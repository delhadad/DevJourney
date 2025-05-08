package controllers

import (
	"fmt"
	"net/http"
	"strings"
	"vendor-management/models"
	"vendor-management/utils"
)

var (
	item_searchCols = []string{
		"items.name",
		"vendors.name",
		"vendors.description",
	}

	// Define the columns you want to select
	item_columns = []string{
		"items.id",
		"items.vendor_id",
		"items.name",
		"items.price",
		"items.created_at",
		"items.updated_at",
		fmt.Sprintf("CASE WHEN NULLIF(items.img, '') IS NOT NULL THEN FORMAT('%s/%%s', items.img) ELSE NULL END AS img", utils.Domain),
		fmt.Sprintf("CASE WHEN NULLIF(vendors.img, '') IS NOT NULL THEN FORMAT('%s/%%s', vendors.img) ELSE NULL END AS \"vendor_img\"", utils.Domain),
		"vendors.id AS \"vendor_id\"",
		"vendors.name AS \"vendor_name\"",
		"vendors.description AS \"vendor_description\"",
		"vendors.created_at AS \"vendor_created_at\"",
		"vendors.updated_at AS \"vendor_updated_at\"",
	}

	// Define the joins
	item_joins = []string{
		"vendors ON items.vendor_id = vendors.id",
	}
)

func IndexItemsHandler(w http.ResponseWriter, r *http.Request) {
	var items []models.Item

	// Generate the SQL query using the generalized BuildQuery function
	meta, err := utils.BuildQuery(&items, "items", item_joins, item_columns, item_searchCols, r.URL.Query(), nil)
	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, fmt.Sprintf("Error building query: %v", err))
		return
	}

	// Send the JSON response
	utils.SendJSONResponse(w, http.StatusOK, models.Response{
		Meta: meta,
		Data: items,
	})
}

func ShowItemHandler(w http.ResponseWriter, r *http.Request) {
	var item models.Item
	id := r.PathValue("id")
	// Create a QB (Query Builder) query
	query, args, err := QB.Select(item_columns...).
		From("items").
		LeftJoin(strings.Join(item_joins, "\nLEFT JOIN ")).
		Where("items.id = ?", id).
		ToSql() // Generate the final SQL query

	if err != nil {
		utils.HandleError(w, http.StatusInternalServerError, "Error building query: "+err.Error())
		return
	}

	// Execute the query to fetch the item
	if err := db.Get(&item, query, args...); err != nil {
		utils.HandleError(w, http.StatusNotFound, fmt.Sprintf("Item not found: %v", err))
		return
	}

	// Send the JSON response
	utils.SendJSONResponse(w, http.StatusOK, item)
}
