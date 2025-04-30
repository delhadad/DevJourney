package utils

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/golang-jwt/jwt/v4"
	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
	"github.com/joho/godotenv"
	"golang.org/x/crypto/bcrypt"
)

var db *sqlx.DB
var (
	Domain    = os.Getenv("DOMAIN")
	secretkey = []byte(os.Getenv("JWT_SECRET"))
)

func SetDB(database *sqlx.DB) {
	db = database
}

// LoadEnv loads environment variables from the .env file
func LoadEnv() error {
	if err := godotenv.Load(); err != nil {
		return err
	}
	return nil
}

// HashPassword hashes a given password
func HashPassword(password string) (string, error) {
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(hashedPassword), nil
}

// HandleError sends a formatted error response
func HandleError(w http.ResponseWriter, status int, message string) {
	w.WriteHeader(status)
	w.Write([]byte(message))
}

// SendJSONResponse sends a JSON response
func SendJSONResponse(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(payload)
}

// SaveImageFile saves an image file to the specified directory
func SaveImageFile(file io.Reader, directory, filename string) (string, error) {
	// Create directory if not exists
	dirPath := filepath.Join("uploads", directory)
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		return "", err
	}

	// Encode filename to make it URL-safe
	encodedFilename := url.PathEscape(filename)

	// Create the file on disk
	filePath := filepath.Join(dirPath, encodedFilename)
	out, err := os.Create(filePath)
	if err != nil {
		return "", err
	}
	defer out.Close()

	// Copy file content
	if _, err := io.Copy(out, file); err != nil {
		return "", err
	}

	return filePath, nil
}

// DeleteImage deletes an image file from disk
func DeleteImage(imagePath *string) error {
	if imagePath != nil {
		if err := os.Remove(*imagePath); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

// GetOptionalString returns a pointer to the string value if it is not empty, otherwise it returns nil.
func GetOptionalString(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

// ! extra
type meta struct {
	Total       int `json:"total"`
	PerPage     int `json:"per_page"`
	CurrentPage int `json:"current_page"`
	FirstPage   int `json:"first_page"`
	LastPage    int `json:"last_page"`
	From        int `json:"from"`
	To          int `json:"to"`
}

func BuildQuery(dest interface{}, table string,
	joins []string, columns []string,
	searchCols []string, queryParams url.Values,
	additionalFilters []string) (*meta, error) {
	q := queryParams.Get("q")
	filters := queryParams.Get("filters")
	sort := queryParams.Get("sort")
	page, _ := strconv.Atoi(queryParams.Get("page"))
	perPage, _ := strconv.Atoi(queryParams.Get("per_page"))

	sb := squirrel.Select().PlaceholderFormat(squirrel.Dollar).From(table)
	for _, join := range joins {
		sb = sb.LeftJoin(join)
	}
	if q != "" {
		orConditions := squirrel.Or{}
		for _, col := range searchCols {
			orConditions = append(orConditions, squirrel.ILike{col: "%" + q + "%"})
		}
		sb = sb.Where(orConditions)
	}

	if filters != "" {
		pairs := strings.Split(filters, ",")
		for _, pair := range pairs {
			parts := strings.Split(pair, ":")
			if len(parts) == 2 {
				sb = sb.Where(squirrel.Eq{parts[0]: parts[1]})
			}
		}
	}

	for _, filter := range additionalFilters {
		sb = sb.Where(filter)
	}
	countSb := sb.Column("COUNT(*)")
	countSQL, countArgs, err := countSb.ToSql()
	if err != nil {
		return nil, err
	}
	var total int
	if err := db.QueryRow(countSQL, countArgs...).Scan(&total); err != nil {
		return nil, err
	}
	sb = sb.Columns(columns...)
	if sort != "" {
		if strings.HasPrefix(sort, "-") {
			sb = sb.OrderBy(strings.TrimPrefix(sort, "-") + " DESC")
		} else {
			sb = sb.OrderBy(sort + " ASC")
		}
	}

	var Offset, lastPage, from, to int
	if page > 0 && perPage > 0 {
		Offset = (page - 1) * perPage
		sb = sb.Limit(uint64(perPage)).Offset(uint64(Offset))
		lastPage = (total + perPage - 1) / perPage
		from = Offset + 1
		to = Offset + perPage
		if to > total {
			to = total
		}
	} else {
		perPage = total
		page = 1
		lastPage = 1
		from = 1
		to = total

	}

	sql, args, err := sb.ToSql()
	if err != nil {
		return nil, err
	}

	if err := db.Select(dest, sql, args...); err != nil {
		return nil, err
	}

	meta := meta{
		Total:       total,
		PerPage:     perPage,
		CurrentPage: page,
		FirstPage:   1,
		LastPage:    lastPage,
		From:        from,
		To:          to,
	}

	return &meta, nil
}

type CustomClaims struct {
	UserID string `json:"user_id"`
	jwt.RegisteredClaims
}
type TokenResponse struct {
	Token     string `json:"token"`
	ExpiresAt string `json:"expires_at"`
}

func GenerateJWT(userID uuid.UUID) (TokenResponse, error) {
	claims := CustomClaims{
		UserID: userID.String(),
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour * 24)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}

	//!print for controlling
	fmt.Println("JWT Secret Key:", secretkey)
	SignedString, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString(secretkey)
	if err != nil {
		return TokenResponse{}, err
	}

	tokenResponse := TokenResponse{
		Token:     SignedString,
		ExpiresAt: fmt.Sprintf("%dh", 24),
	}
	return tokenResponse, nil
}

func ParseJWT(tokenString string) (uuid.UUID, error) {
	token, err := jwt.ParseWithClaims(tokenString, &CustomClaims{}, func(token *jwt.Token) (interface{}, error) {
		return secretkey, nil
	})
	if err != nil {
		return uuid.Nil, err
	}
	claims, ok := token.Claims.(*CustomClaims)
	if !ok || !token.Valid {
		return uuid.Nil, errors.New("invalid token")
	}

	userID, err := uuid.Parse(claims.UserID)
	if err != nil {
		return uuid.Nil, err
	}
	return userID, nil
}
