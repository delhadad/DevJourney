package controllers

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
	"vendor-management/models"
	"vendor-management/utils"

	"github.com/Masterminds/squirrel"
	"github.com/google/uuid"
)

var (
	vendor_columns = []string{
		"id",
		"name",
		"description",
		"created_at",
		"updated_at",
		fmt.Sprintf("CASE WHEN NULLIF(img, '') IS NOT NULL THEN FORMAT('%s/%%s', img) ELSE NULL END AS img", Domain),
	}
)

/*
	 func IndexVendorsHandler(w http.ResponseWriter, r *http.Request) {
		var vendors []models.Vendor

		// Use the BuildQuery function to construct the query with pagination and metadata
		meta, err := utils.BuildQuery(
			&vendors,
			"vendors",
			nil,
			vendor_columns,
			nil,
			r.URL.Query(),
			nil,
		)

		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, fmt.Sprintf("Error building query: %v", err))
			return
		}

		// Send the response with the meta information and the data
		utils.SendJSONResponse(w, http.StatusOK, models.Response{
			Meta: meta,
			Data: vendors,
		})
	}
*/
func IndexVendorsHandler(w http.ResponseWriter, r *http.Request) {
	var vendors []models.Vendor

	page, err := strconv.Atoi(r.URL.Query().Get("page"))
	if err != nil || page < 1 {
		page = 1
	}

	pageSize := 10

	// Extract the search query parameter
	searchTerm := r.URL.Query().Get("search")

	// Start building the query
	queryBuilder := QB.Select(strings.Join(vendor_columns, ", ")).
		From("vendors").
		Limit(uint64(pageSize)).
		Offset(uint64((page - 1) * pageSize))

	// If search term is provided, add case-insensitive search condition using LOWER()
	if searchTerm != "" {
		searchTerm = "%" + strings.ToLower(searchTerm) + "%" // Convert the search term to lowercase
		queryBuilder = queryBuilder.Where(
			squirrel.Or{
				squirrel.Expr("LOWER(name) LIKE ?", searchTerm),
				squirrel.Expr("LOWER(description) LIKE ?", searchTerm),
			},
		)
	}

	// Execute the query
	query, args, err := queryBuilder.ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Fetch the vendors from the database
	if err := db.Select(&vendors, query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Count the total number of vendors
	var totalVendors int
	err = db.Get(&totalVendors, "SELECT COUNT(*) FROM vendors")
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}

	response := map[string]interface{}{
		"meta": map[string]interface{}{
			"current_page": page,
			"per_page":     pageSize,
			"total":        totalVendors,
		},
		"data": vendors,
	}

	utils.SendJSONResponse(w, http.StatusOK, response)
}

func ShowVendorHandler(w http.ResponseWriter, r *http.Request) {
	var vendor models.Vendor
	id := r.PathValue("id")

	query, args, err := QB.Select(strings.Join(vendor_columns, ", ")).From("vendors").From("vendors").Where("id = ?", id).ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := db.Get(&vendor, query, args...); err != nil {
		utils.HandleError(w, http.StatusNotFound, "Vendor not found")
		return
	}
	utils.SendJSONResponse(w, http.StatusOK, vendor)
}

func CreateVendorHandler(w http.ResponseWriter, r *http.Request) {
	vendor := models.Vendor{
		ID:          uuid.New(),
		Name:        r.FormValue("name"),
		Description: utils.GetOptionalString(r.FormValue("description")),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	file, fileHeader, err := r.FormFile("img")
	if err != nil && err != http.ErrMissingFile {
		utils.HandleError(w, http.StatusBadRequest, "Invalid file")
		return
	} else if err == nil {
		defer file.Close()
		imageName, err := utils.SaveImageFile(file, "vendors", fileHeader.Filename)
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error saving image")
			return
		}
		vendor.Img = &imageName
	}

	query, args, err := QB.
		Insert("vendors").
		Columns("id", "name", "img", "description", "created_at", "updated_at").
		Values(vendor.ID, vendor.Name, vendor.Img, vendor.Description, vendor.CreatedAt, vendor.UpdatedAt).
		Suffix(fmt.Sprintf("RETURNING %s", strings.Join(vendor_columns, ", "))).
		ToSql()

	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error generating query")
		return
	}

	if err := db.QueryRowx(query, args...).StructScan(&vendor); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error creating vendor: "+err.Error())
		return
	}

	utils.SendJSONResponse(w, http.StatusCreated, vendor)
}

func UpdateVendorHandler(w http.ResponseWriter, r *http.Request) {
	var vendor models.Vendor
	id := r.PathValue("id")

	// Fetch the existing vendor
	query, args, err := QB.Select(strings.Join(vendor_columns, ", ")).From("vendors").Where("id = ?", id).ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := db.Get(&vendor, query, args...); err != nil {
		utils.HandleError(w, http.StatusNotFound, "Vendor not found")
		return
	}

	// Update vendor details
	if r.FormValue("name") != "" {
		vendor.Name = r.FormValue("name")
	}
	if r.FormValue("description") != "" {
		vendor.Description = utils.GetOptionalString(r.FormValue("description"))
	}

	// Handle image file update
	file, fileHeader, err := r.FormFile("img")
	if err != nil && err != http.ErrMissingFile {
		utils.HandleError(w, http.StatusBadRequest, "Invalid file")
		return
	} else if err == nil {
		defer file.Close()
		imageName, err := utils.SaveImageFile(file, "vendors", fileHeader.Filename)
		if err != nil {
			utils.HandleError(w, http.StatusInternalServerError, "Error saving image")
			return
		}
		if vendor.Img != nil {
			// Delete old image
			utils.DeleteImage(vendor.Img)
		}
		vendor.Img = &imageName
	}

	// Update the vendor in the database
	query, args, err = QB.
		Update("vendors").
		Set("name", vendor.Name).
		Set("img", vendor.Img).
		Set("description", vendor.Description).
		Set("updated_at", time.Now()).
		Where("id = ?", vendor.ID).
		Suffix("RETURNING *").
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if err := db.QueryRowx(query, args...).StructScan(&vendor); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error updating vendor: "+err.Error())
		return
	}

	utils.SendJSONResponse(w, http.StatusOK, vendor)
}

// Delete a vendor
func DeleteVendorHandler(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	// Delete vendor from database
	query, args, err := QB.Delete("vendors").
		Where("id = ?", id).
		Suffix("RETURNING img").
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query: "+err.Error())
		return
	}

	var img *string
	if err := db.QueryRow(query, args...).Scan(&img); err != nil {
		utils.HandleError(w, http.StatusNotFound, "Vendor not found")
		return
	}

	// Delete old image
	utils.DeleteImage(img)

	utils.SendJSONResponse(w, http.StatusOK, "Vendor deleted")
}

func AssignVendorAdminHandler(w http.ResponseWriter, r *http.Request) {
	// Extract user_id and vendor_id from form data
	userID := r.FormValue("user_id")
	vendorID := r.FormValue("vendor_id")

	// Ensure both user_id and vendor_id are provided
	if userID == "" || vendorID == "" {
		utils.HandleError(w, http.StatusBadRequest, "user_id and vendor_id are required")
		return
	}

	// Convert strings to UUID
	userUUID, err := uuid.Parse(userID)
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid user_id format")
		return
	}
	vendorUUID, err := uuid.Parse(vendorID)
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid vendor_id format")
		return
	}

	// Insert the new admin assignment into vendor_admins table
	query, args, err := QB.Insert("vendor_admins").
		Columns("vendor_id", "user_id").
		Values(vendorUUID, userUUID).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if _, err := db.Exec(query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error assigning vendor admin")
		return
	}

	utils.SendJSONResponse(w, http.StatusOK, "Admin assigned successfully")
}

func RevokeVendorAdminHandler(w http.ResponseWriter, r *http.Request) {
	// Extract user_id and vendor_id from form data
	userID := r.FormValue("user_id")
	vendorID := r.FormValue("vendor_id")

	// Ensure both user_id and vendor_id are provided
	if userID == "" || vendorID == "" {
		utils.HandleError(w, http.StatusBadRequest, "user_id and vendor_id are required")
		return
	}

	// Convert strings to UUID
	userUUID, err := uuid.Parse(userID)
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid user_id format")
		return
	}
	vendorUUID, err := uuid.Parse(vendorID)
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid vendor_id format")
		return
	}

	// Delete the admin assignment from vendor_admins table
	query, args, err := QB.Delete("vendor_admins").
		Where(squirrel.Eq{"vendor_id": vendorUUID, "user_id": userUUID}).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if _, err := db.Exec(query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error revoking vendor admin")
		return
	}

	utils.SendJSONResponse(w, http.StatusOK, "Admin revoked successfully")
}

func IndexVendorAdminsHandler(w http.ResponseWriter, r *http.Request) {
	vendorID := r.PathValue("vendor_id")

	// Ensure vendor_id is provided
	if vendorID == "" {
		utils.HandleError(w, http.StatusBadRequest, "vendor_id is required")
		return
	}

	// Convert vendorID to UUID
	vendorUUID, err := uuid.Parse(vendorID)
	if err != nil {
		utils.HandleError(w, http.StatusBadRequest, "Invalid vendor_id format")
		return
	}

	// Query to get all admins for the given vendor
	var admins []models.User
	query, args, err := QB.Select("users.id", "users.name", "users.email").
		From("vendor_admins").
		LeftJoin("users ON vendor_admins.user_id = users.id").
		Where(squirrel.Eq{"vendor_admins.vendor_id": vendorUUID}).
		ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if err := db.Select(&admins, query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error fetching vendor admins")
		return
	}

	utils.SendJSONResponse(w, http.StatusOK, admins)
}
