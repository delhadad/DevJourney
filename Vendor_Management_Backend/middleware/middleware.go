package middleware

import (
	"context"
	"net/http"
	"strings"
	"vendor-management/controllers"
	"vendor-management/models"
	"vendor-management/utils"

	"github.com/jmoiron/sqlx"
)

func JWTMiddleware(db *sqlx.DB) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var accessToken string

			cookie, err := r.Cookie("accessToken")
			if err == nil {
				accessToken = strings.Replace(cookie.Value, "accessToken", "", 1)
			}

			bearer := r.Header.Get("Authorization")
			if bearer != "" {
				splitBearer := strings.Split(bearer, " ")
				if len(splitBearer) == 2 {
					accessToken = splitBearer[1]
				}
			}

			if accessToken == "" {
				next.ServeHTTP(w, r)
				return
			}

			userID, err := utils.ParseJWT(accessToken)
			if err != nil {
				utils.HandleError(w, http.StatusUnauthorized, "Invalid access token")
				return
			}
			//
			var user models.User
			if err := db.Get(&user, "SELECT * FROM users WHERE id = $1", userID); err != nil {
				utils.HandleError(w, http.StatusUnauthorized, "User not found")
				return
			}

			if err := controllers.GetRoles(&user); err != nil {
				utils.HandleError(w, http.StatusInternalServerError, "Unable to fetch user roles")
			}

			ctx := context.WithValue(r.Context(), "user", user)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func RoleMiddleware(allowedRoles ...int) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

			user, ok := r.Context().Value("user").(models.User)
			if !ok {
				utils.HandleError(w, http.StatusUnauthorized, "Unauthorized: User information is missing")
				return // Return early if user is not authenticated or not present in the context.Value() map.                }
			}
			roleMap := make(map[int]bool)
			for _, role := range user.Roles {
				roleMap[role] = true
			}

			for _, allowedRole := range allowedRoles {
				if roleMap[allowedRole] {
					next.ServeHTTP(w, r)
					return
				}
			}

			utils.HandleError(w, http.StatusForbidden, "Forbidden: You do not have the requiredn role to access this resource")
		})
	}
}
