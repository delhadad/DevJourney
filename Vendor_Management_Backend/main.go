package main

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path"
	"vendor-management/controllers"
	"vendor-management/middleware"
	"vendor-management/utils"

	"github.com/go-michi/michi"
	"github.com/golang-migrate/migrate/v4"
	_ "github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	"github.com/jmoiron/sqlx"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

func main() {

	controllers.InitMqttClient()

	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Fatal(err)
	}

	db, err := sqlx.Connect("postgres", os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Run migrations
	migrationPath := "file://" + GetRootpath("database/migrations")
	mig, err := migrate.New(
		migrationPath,
		os.Getenv("DATABASE_URL"),
	)
	if err != nil {
		log.Fatal(err)
	}

	if err := mig.Up(); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		log.Fatal(err)
	}
	if err == migrate.ErrNoChange {
		log.Printf("Migration: %s", err.Error())
	}

	// Pass the database connection to controllers
	controllers.SetDB(db)
	utils.SetDB(db)

	// Initialize Michi Router
	r := michi.NewRouter()
	r.Use(middleware.CORSMiddleware)
	r.Handle("/uploads/", http.StripPrefix("/uploads/", http.FileServer(http.Dir("uploads"))))
	r.Route("/", func(sub *michi.Router) {
		sub.Use(middleware.JWTMiddleware(db))
		// Static file handler for uploads

		sub.HandleFunc("POST signup", controllers.SignUpHandler)
		sub.HandleFunc("POST login", controllers.LoginHandler)
		sub.HandleFunc("GET me", controllers.MeHandler)
		sub.HandleFunc("PUT me", controllers.UpdateMeHandler)

		sub.Group(func(userApi *michi.Router) {
			userApi.Use(middleware.RoleMiddleware(1))
			userApi.HandleFunc("GET users", controllers.IndexUsersHandler)
			userApi.HandleFunc("GET users/{id}", controllers.ShowUsersHandler)
			userApi.HandleFunc("PUT users/{id}", controllers.UpdateUsersHandler)
			userApi.HandleFunc("DELETE users/{id}", controllers.DeleteUsersHandler)
		})

		// Vendor routes
		sub.HandleFunc("GET vendors", controllers.IndexVendorsHandler)
		sub.HandleFunc("GET vendors/{id}", controllers.ShowVendorHandler)
		sub.HandleFunc("POST vendors", controllers.CreateVendorHandler)
		sub.HandleFunc("PUT vendors/{id}", controllers.UpdateVendorHandler)
		sub.HandleFunc("DELETE vendors/{id}", controllers.DeleteVendorHandler)
		sub.HandleFunc("POST vendors/assign-admin", controllers.AssignVendorAdminHandler)
		sub.HandleFunc("POST vendors/revoke-admin", controllers.RevokeVendorAdminHandler)
		sub.HandleFunc("GET vendors/{vendor_id}/admins", controllers.IndexVendorAdminsHandler)

		// tables routes
		sub.HandleFunc("GET tables", controllers.IndexTablesHandler)
		sub.HandleFunc("GET tables/{table_id}", controllers.ShowTableHandler)
		sub.HandleFunc("POST tables", controllers.StoreTableHandler)
		sub.HandleFunc("PUT tables/{table_id}", controllers.UpdateTableHandler)
		sub.HandleFunc("DELETE tables/{table_id}", controllers.DeleteTableHandler)

		/* 		// items routes
		   		sub.HandleFunc("GET items", controllers.IndexItemsHandler)
		   		sub.HandleFunc("GET items/{id}", controllers.ShowItemHandler)
		   		sub.Group(func(VendorApi *michi.Router) {
		   			itemApi.Use(middleware.RoleMiddleware(1, 2))

		   			itemApi.HandleFunc("POST items", controllers.CreateItemHandler)
		   			itemApi.HandleFunc("PUT items/{id}", controllers.UpdateItemHandler)
		   			itemApi.HandleFunc("DELETE items/{id}", controllers.DeleteItemHandler)
		   		})  */

		sub.HandleFunc("POST led/on", controllers.TurnLedOn)
		sub.HandleFunc("POST led/off", controllers.TurnLedOff)

	})

	// Start the server
	fmt.Println("Starting server on port 8080")
	if err := http.ListenAndServe(":8080", r); err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

// GetRootpath gets the absolute path for a given directory
func GetRootpath(dir string) string {
	ex, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	return path.Join(path.Dir(ex), dir)
}
