package controllers

import (
	"database/sql"
	"fmt"
	"net/http"
	"vendor-management/models"
	"vendor-management/utils"

	"github.com/Masterminds/squirrel"
	"github.com/google/uuid"
)

func IndexTablesHandler(w http.ResponseWriter, r *http.Request) {
	var tables []models.Table

	query, args, err := QB.Select(`
        tables.id, tables.vendor_id, tables.name, tables.customer_id, tables.is_available, tables.is_needs_service,
        vendors.id AS "vendor.id", vendors.name AS "vendor.name", vendors.img AS "vendor.img",
        vendors.description AS "vendor.description", vendors.created_at AS "vendor.created_at", vendors.updated_at AS "vendor.updated_at",
        users.id AS "customer.id", users.name AS "customer.name", users.phone AS "customer.phone", users.email AS "customer.email", users.img AS "customer.img"
    `).
		From("tables").
		LeftJoin("vendors ON vendors.id = tables.vendor_id").
		LeftJoin("users ON users.id = tables.customer_id").
		ToSql()

	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	// Execute the query
	if err := db.Select(&tables, query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error fetching tables")
		return
	}

	// Loop through each table and modify image URLs
	for i := range tables {
		// Construct full URL for vendor image
		if tables[i].Vendor.Img != nil {
			imgURL := fmt.Sprintf("%s/%s", utils.Domain, *tables[i].Vendor.Img)
			tables[i].Vendor.Img = &imgURL
		}

		// Construct full URL for customer image if it exists
		if tables[i].Customer != nil && tables[i].Customer.Img != nil {
			imgURL := fmt.Sprintf("%s/%s", utils.Domain, *tables[i].Customer.Img)
			tables[i].Customer.Img = &imgURL
		}
	}

	utils.SendJSONResponse(w, http.StatusOK, tables)
}

func ShowTableHandler(w http.ResponseWriter, r *http.Request) {
	tableID := r.PathValue("table_id")

	query, args, err := QB.Select(`
        tables.id, tables.vendor_id, tables.name, tables.customer_id, tables.is_available, tables.is_needs_service,
        vendors.id AS "vendor.id", vendors.name AS "vendor.name", vendors.img AS "vendor.img",
        vendors.description AS "vendor.description", vendors.created_at AS "vendor.created_at", vendors.updated_at AS "vendor.updated_at",
        users.id AS "customer.id", users.name AS "customer.name", users.phone AS "customer.phone", users.email AS "customer.email", users.img AS "customer.img"
    `).
		From("tables").
		LeftJoin("vendors ON vendors.id = tables.vendor_id").
		LeftJoin("users ON users.id = tables.customer_id").
		Where("tables.id = ?", tableID).
		ToSql()

	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	var table models.Table
	if err := db.Get(&table, query, args...); err != nil {
		if err == sql.ErrNoRows {
			utils.HandleError(w, http.StatusNotFound, "Table not found")
		} else {
			utils.HandleError(w, http.StatusInternalServerError, err.Error())
		}
		return
	}

	// Construct full URL for vendor image
	if table.Vendor.Img != nil {
		imgURL := fmt.Sprintf("%s/%s", utils.Domain, *table.Vendor.Img)
		table.Vendor.Img = &imgURL
	}

	// Construct full URL for customer image if it exists
	if table.Customer != nil && table.Customer.Img != nil {
		imgURL := fmt.Sprintf("%s/%s", utils.Domain, *table.Customer.Img)
		table.Customer.Img = &imgURL
	}

	utils.SendJSONResponse(w, http.StatusOK, table)
}

// StoreTableHandler - Create a new table
func StoreTableHandler(w http.ResponseWriter, r *http.Request) {
	// Read the form data
	var table models.Table
	table.Name = r.FormValue("name")
	table.VendorID = uuid.MustParse(r.FormValue("vendor_id"))
	table.CustomerID = nil // Optional, can be set later

	// Insert new table into the database
	query, args, err := QB.Insert("tables").
		Columns("name", "vendor_id", "is_available", "is_needs_service").
		Values(table.Name, table.VendorID, true, false).
		Suffix("RETURNING id, name, is_available, is_needs_service").
		ToSql()

	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building insert query")
		return
	}

	// Execute the query and return the newly created table
	if err := db.QueryRowx(query, args...).StructScan(&table); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error inserting table")
		return
	}

	utils.SendJSONResponse(w, http.StatusCreated, table)
}

// UpdateTableHandler - Update table details
func UpdateTableHandler(w http.ResponseWriter, r *http.Request) {
	tableID := r.PathValue("table_id")
	var table models.Table

	// Fetch the existing table record
	query, args, err := QB.Select("*").From("tables").Where(squirrel.Eq{"id": tableID}).ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		fmt.Println("Error building select query:", err)
		return
	}

	if err := db.Get(&table, query, args...); err != nil {
		utils.HandleError(w, http.StatusNotFound, "Table not found")
		fmt.Println("Error fetching table:", err)
		return
	}

	// Update the fields based on the input from the request
	if r.FormValue("name") != "" {
		table.Name = r.FormValue("name")
	}

	if r.FormValue("is_available") != "" {
		table.IsAvailable = r.FormValue("is_available") == "true"
	}

	// Build update SQL query without updated_at
	query, args, err = QB.Update("tables").
		Set("name", table.Name).
		Set("is_available", table.IsAvailable).
		Where(squirrel.Eq{"id": table.ID}).ToSql()

	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building update query")
		fmt.Println("Error building update query:", err)
		return
	}

	// Execute the update query
	if _, err := db.Exec(query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error updating table")
		fmt.Println("Database execution error:", err)
		return
	}

	fmt.Println("Table updated successfully")
	utils.SendJSONResponse(w, http.StatusOK, table)
}

// DeleteTableHandler - Delete a table by ID
func DeleteTableHandler(w http.ResponseWriter, r *http.Request) {
	tableID := r.PathValue("table_id")

	query, args, err := QB.Delete("tables").Where(squirrel.Eq{"id": tableID}).ToSql()
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query")
		return
	}

	if _, err := db.Exec(query, args...); err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error deleting table")
		return
	}
	utils.SendJSONResponse(w, http.StatusOK, "Table deleted successfully")
}
