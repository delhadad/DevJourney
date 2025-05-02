package models

import (
	"time"

	"github.com/google/uuid"
)

type User struct {
	ID        uuid.UUID `db:"id" json:"id"`
	Name      string    `db:"name" json:"name"`
	Email     string    `db:"email" json:"email"`
	Phone     string    `db:"phone" json:"phone"`
	Password  string    `db:"password" json:"-"`
	Img       *string   `db:"img" json:"img,omitempty"`
	CreatedAt time.Time `db:"created_at" json:"created_at"`
	UpdatedAt time.Time `db:"updated_at" json:"updated_at"`

	Roles []int `db:"roles" json:"roles"`
}

type Response struct {
	Meta interface{} `json:"meta"`
	Data interface{} `json:"data"`
}
