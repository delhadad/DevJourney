package controllers

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
	"vendor-management/models"
	"vendor-management/utils"

	"github.com/Masterminds/squirrel"
	"github.com/jmoiron/sqlx"
	_ "github.com/joho/godotenv/autoload"
)

var (
	db *sqlx.DB
	QB = squirrel.StatementBuilder.PlaceholderFormat(squirrel.Dollar)
)

func SetDB(database *sqlx.DB) {
	db = database
}

var (
	Domain       = os.Getenv("DOMAIN")
	user_columns = []string{
		"id",
		"name",
		"email",
		"phone",
		"created_at",
		"updated_at",
		fmt.Sprintf("CASE WHEN NULLIF(img, '') IS NOT NULL THEN FORMAT('%s/%%s', img) ELSE NULL END AS img", Domain),
	}

	user_searchCola = []string{
		"name",
		"email",
		"phone"}
)

func IndexUsersHandler(w http.ResponseWriter, r *http.Request) {
	var users []models.User
	meta, err := utils.BuildQuery(&users, "users",
		nil, user_columns, user_searchCola,
		r.URL.Query(), nil)
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, fmt.Sprintf("Error building query: %v", err))
		return
	}

	// Fetch and attach roles for each user
	for i := range users {
		err := GetRoles(&users[i])
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error fetching user roles: "+err.Error())
			return
		}
	}

	utils.SendJSONResponse(w, http.StatusOK,
		models.Response{
			Meta: meta,
			Data: users,
		})
}
func ShowUsersHandler(w http.ResponseWriter, r *http.Request) {
	var user models.User
	id := r.PathValue("id")

	query, args, err := QB.Select(strings.Join(user_columns, ", ")).
		From("users").
		Where("id = ?", id).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := db.Get(&user, query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Fetch and attach roles
	err = GetRoles(&user)
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error fetching user roles: "+err.Error())
		return
	}

	utils.SendJSONResponse(w, http.StatusOK, user)
}

func UpdateUsersHandler(w http.ResponseWriter, r *http.Request) {
	var user models.User
	id := r.PathValue("id")
	query, args, err := QB.Select("id", "name", "phone", "email", "password", "img").
		From("users").
		Where("id = ?", id).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := db.Get(&user, query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Update user details
	if r.FormValue("name") != "" {
		user.Name = r.FormValue("name")
	}
	if r.FormValue("phone") != "" {
		user.Phone = r.FormValue("phone")
	}
	if r.FormValue("email") != "" {
		user.Email = r.FormValue("email")
	}
	if r.FormValue("password") != "" {
		hashedPassword, err := utils.HashPassword(r.FormValue("password"))
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error hashing password")
			return
		}
		user.Password = hashedPassword
	}

	// Handle image file update
	file, fileHeader, err := r.FormFile("img")
	if err != nil && err != http.ErrMissingFile {
		utils.HandleError(w, http.StatusBadRequest, "Invalid file")
		return
	} else if err == nil {
		defer file.Close()
		imageName, err := utils.SaveImageFile(file, "users", fileHeader.Filename)
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error saving image")
		}
		if user.Img != nil {
			// Delete old image
			utils.DeleteImage(user.Img)
		}
		user.Img = &imageName
	}

	// Update user in the database
	query, args, err = QB.
		Update("users").
		Set("img", user.Img).
		Set("name", user.Name).
		Set("email", user.Email).
		Set("phone", user.Phone).
		Set("password", user.Password).
		Set("updated_at", time.Now()).
		Where(squirrel.Eq{"id": user.ID}).
		Suffix(fmt.Sprintf("RETURNING %s", strings.Join(user_columns, ", "))).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if err := db.QueryRowx(query, args...).StructScan(&user); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error updating user: "+err.Error())
		return
	}

	utils.SendJSONResponse(w, http.StatusOK, user)
}

func DeleteUsersHandler(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	// Delete user from database
	query, args, err := QB.Delete("users").
		Where("id = ?", id).
		Suffix("RETURNING img").
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query: "+err.Error())
		return
	}

	var img *string
	if err := db.QueryRow(query, args...).Scan(&img); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error deleting user: "+err.Error())
		return
	}

	// Delete old image
	utils.DeleteImage(img)

	utils.SendJSONResponse(w, http.StatusOK, "User deleted")
}

func GetRoles(user *models.User) error {
	user.Roles = []int{}

	query, args, err := QB.Select("user_roles.role_id").
		From("user_roles").
		LeftJoin("users ON user_roles.user_id = users.id").
		Where(squirrel.Eq{"users.id": user.ID}).
		ToSql()

	if err != nil {
		return err
	}

	err = db.SelectContext(context.Background(), &user.Roles, query, args...)
	if err != nil {
		return err
	}

	return nil
}

func MeHandler(w http.ResponseWriter, r *http.Request) {
	// Get the user from the request context (set by JWTMiddleware)
	user, ok := r.Context().Value("user").(models.User)
	if !ok {
		utils.HandleError(w, http.StatusUnauthorized, "Unauthorized: User not found in context")
		return
	}

	// Ensure the image URL is complete
	if user.Img != nil && *user.Img != "" {
		imgURL := fmt.Sprintf("%s/%s", utils.Domain, *user.Img)
		user.Img = &imgURL
	}

	// Send the current user details as a response
	utils.SendJSONResponse(w, http.StatusOK, user)
}

func UpdateMeHandler(w http.ResponseWriter, r *http.Request) {
	// Get the user from the request context (set by JWTMiddleware)
	user, ok := r.Context().Value("user").(models.User)
	if !ok {
		utils.HandleError(w, http.StatusUnauthorized, "Unauthorized: User not found in context")
		return
	}

	// Update user details from form data (allow changes to name, email, phone, password)
	if r.FormValue("name") != "" {
		user.Name = r.FormValue("name")
	}
	if r.FormValue("phone") != "" {
		user.Phone = r.FormValue("phone")
	}
	if r.FormValue("email") != "" {
		user.Email = r.FormValue("email")
	}
	if r.FormValue("password") != "" {
		hashedPassword, err := utils.HashPassword(r.FormValue("password"))
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error hashing password")
			return
		}
		user.Password = hashedPassword
	}

	// Handle image file update
	file, fileHeader, err := r.FormFile("img")
	if err != nil && err != http.ErrMissingFile {
		utils.HandleError(w, http.StatusBadRequest, "Invalid file")
		return
	} else if err == nil {
		defer file.Close()
		imageName, err := utils.SaveImageFile(file, "users", fileHeader.Filename)
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error saving image")
			return
		}
		if user.Img != nil {
			// Delete old image
			utils.DeleteImage(user.Img)
		}
		user.Img = &imageName
	}

	// Update the user in the database
	query, args, err := QB.
		Update("users").
		Set("name", user.Name).
		Set("email", user.Email).
		Set("phone", user.Phone).
		Set("password", user.Password).
		Set("img", user.Img).
		Set("updated_at", time.Now()).
		Where(squirrel.Eq{"id": user.ID}).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if _, err := db.Exec(query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error updating user: "+err.Error())
		return
	}

	// Ensure the image URL is complete
	if user.Img != nil && *user.Img != "" {
		imgURL := fmt.Sprintf("%s/%s", utils.Domain, *user.Img)
		user.Img = &imgURL
	}

	// Send the updated user details as a response
	utils.SendJSONResponse(w, http.StatusOK, user)
}
