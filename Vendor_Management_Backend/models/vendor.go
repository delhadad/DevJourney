package models

import (
	"time"

	"github.com/google/uuid"
)

type Vendor struct {
	ID          uuid.UUID `db:"id" json:"id"`
	Name        string    `db:"name" json:"name"`
	Img         *string   `db:"img" json:"img,omitempty"`
	Description *string   `db:"description" json:"description,omitempty"`
	CreatedAt   time.Time `db:"created_at" json:"created_at"`
	UpdatedAt   time.Time `db:"updated_at" json:"updated_at"`
}

type VendorAdmin struct {
	VendorID uuid.UUID `db:"vendor_id" json:"vendor_id"`
	UserID   uuid.UUID `db:"user_id" json:"user_id"`
}
