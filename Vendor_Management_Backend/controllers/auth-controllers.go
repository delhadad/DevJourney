package controllers

import (
	"database/sql"
	"fmt"
	"net/http"
	"strings"
	"time"
	"vendor-management/models"
	"vendor-management/utils"

	"github.com/Masterminds/squirrel"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

// SignUpHandler handles user registration (signup)
func SignUpHandler(w http.ResponseWriter, r *http.Request) {
	user := models.User{
		ID:        uuid.New(),
		Name:      r.FormValue("name"),
		Phone:     r.FormValue("phone"),
		Email:     r.FormValue("email"),
		Password:  r.FormValue("password"),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if user.Password == "" {
		utils.HandleError(w, http.StatusBadRequest, "Password is required")
		return
	}

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
		user.Img = &imageName
	}

	hashedPassword, err := utils.HashPassword(user.Password)
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error hashing password")
		return
	}
	user.Password = hashedPassword

	query, args, err := QB.
		Insert("users").
		Columns("id", "img", "name", "phone", "email", "password").
		Values(user.ID, user.Img, user.Name, user.Phone, user.Email, user.Password).
		Suffix(fmt.Sprintf("RETURNING %s", strings.Join(user_columns, ", "))).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error generating query")
		return
	}

	if err := db.QueryRowx(query, args...).StructScan(&user); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error creating user: "+err.Error())
		return
	}

	utils.SendJSONResponse(w, http.StatusCreated, user)
}

// LoginHandler handles user authentication (login)
/* func LoginHandler(w http.ResponseWriter, r *http.Request) {
	// Parse form values
	if err := r.ParseForm(); err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid form data")
		return
	}

	var loginReq LoginRequest
	if err := decoder.Decode(&loginReq, r.PostForm); err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid login credentials")
		return
	}

	// Find user by email
	var user models.User
	query := "SELECT id, name, email, phone, password FROM users WHERE email = $1"
	if err := db.Get(&user, query, loginReq.Email); err != nil {
		utils.HandleError(w, http.StatusUnauthorized, "Invalid email or password")
		return
	}

	// Compare hashed password with the plain text password
	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(loginReq.Password)); err != nil {
		utils.HandleError(w, http.StatusUnauthorized, "Invalid email or password")
		return
	}

	// Return a success response
	utils.SendJSONResponse(w, http.StatusOK, map[string]interface{}{
		"message": "Login successful",
		"user":    user,
	})
} */
/* func LoginHandler(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form
	err := r.ParseMultipartForm(32 << 20) // 32MB is the max memory size for form parsing
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid form data")
		return
	}

	// Extract form values
	email := r.FormValue("email")
	password := r.FormValue("password")

	if email == "" || password == "" {
		utils.HandleError(w, http.StatusBadRequest, "Email and password are required")
		return
	}

	// Find user by email
	var user models.User
	query := "SELECT id, name, email, phone, password FROM users WHERE email = $1"
	if err := db.Get(&user, query, email); err != nil {
		utils.HandleError(w, http.StatusUnauthorized, "Invalid email or password")
		return
	}

	// Compare hashed password with the plain text password
	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		utils.HandleError(w, http.StatusUnauthorized, "Invalid email or password")
		return
	}

	// Return a success response
	utils.SendJSONResponse(w, http.StatusOK, map[string]interface{}{
		"message": "Login successful",
		"user":    user,
	})
}
*/
/* type Claims struct {
	Email string `json:"email"`
	ID    string `json:"id"`
	jwt.StandardClaims
}

func LoginHandler(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form
	err := r.ParseMultipartForm(32 << 20) // 32MB is the max memory size for form parsing
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid form data")
		return
	}

	// Extract form values
	email := r.FormValue("email")
	password := r.FormValue("password")

	if email == "" || password == "" {
		utils.HandleError(w, http.StatusBadRequest, "Email and password are required")
		return
	}

	// Find user by email
	var user models.User
	query := "SELECT id, name, email, phone, password FROM users WHERE email = $1"
	if err := db.Get(&user, query, email); err != nil {
		utils.HandleError(w, http.StatusUnauthorized, "Invalid email or password")
		return
	}

	// Compare hashed password with the plain text password
	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		utils.HandleError(w, http.StatusUnauthorized, "Invalid email or password")
		return
	}

	// Create JWT token for the authenticated user
	expirationTime := time.Now().Add(24 * time.Hour) // Token expires after 24 hours
	claims := &Claims{
		Email: user.Email,
		ID:    user.ID.String(), // Assuming user.ID is a UUID
		StandardClaims: jwt.StandardClaims{
			ExpiresAt: expirationTime.Unix(),
		},
	}

	// Create the JWT
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(jwtKey)
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error generating token")
		return
	}

	// Return the token and user info in the response
	utils.SendJSONResponse(w, http.StatusOK, map[string]interface{}{
		"message": "Login successful",
		"user":    user,
		"token":   tokenString,
	})
} */
//! last one
func LoginHandler(w http.ResponseWriter, r *http.Request) {
	var credentials struct {
		Email    string
		Password string
	}

	credentials.Email = r.FormValue("email")
	credentials.Password = r.FormValue("password")

	var user models.User
	query, args, err := QB.Select("id", "password").
		From("users").
		Where(squirrel.Eq{"email": credentials.Email}).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if err := db.Get(&user, query, args...); err != nil {
		if err == sql.ErrNoRows {
			utils.HandleError(w, http.StatusNotFound, "User not found")
			return
		}
		utils.HandleError(w, http.StatusInternalServerError, "Error featching user")
		return
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(credentials.Password)); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Invalid credentials")
		return
	}
	//!print
	fmt.Println(user.ID)

	tokenRespone, err := utils.GenerateJWT(user.ID)
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error generating JWT")
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     "accessToken",
		Value:    tokenRespone.Token,
		Path:     "/",
		Expires:  time.Now().UTC().Add(time.Hour * 24),
		HttpOnly: true,
		Secure:   false,
		SameSite: http.SameSiteNoneMode,
	})
	utils.SendJSONResponse(w, http.StatusOK, tokenRespone)
}
