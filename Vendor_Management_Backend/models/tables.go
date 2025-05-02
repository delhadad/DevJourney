/* package models

import (

	"github.com/google/uuid"
)

type Table struct {
	ID             uuid.UUID  `db:"id" json:"id"`
	VendorID       uuid.UUID  `db:"vendor_id" json:"vendor_id"` // Fixed typo in json field
	Name           string     `db:"name" json:"name"`
	CustomerID     *uuid.UUID `db:"customer_id" json:"customer_id"`
	IsAvailable    bool       `db:"is_available" json:"is_available"`
	IsNeedsService bool       `db:"is_needs_service" json:"is_needs_service"`

	Vendor   *Vendor `json:"vendor,omitempty"`   // Embedded vendor details
	Customer *User   `json:"customer,omitempty"` // Embedded customer details
}
*/

package models

import (
	"github.com/google/uuid"
)

// Import the Vendor struct from vendor.go
// Assume it's in the same package

type Customer struct {
	ID    *uuid.UUID `db:"id" json:"id,omitempty"` // Handle nullable UUIDs
	Name  *string    `db:"name" json:"name,omitempty"`
	Phone *string    `db:"phone" json:"phone,omitempty"`
	Email *string    `db:"email" json:"email,omitempty"`
	Img   *string    `db:"img" json:"img,omitempty"`
}
type Table struct {
	ID             uuid.UUID  `db:"id" json:"id"`
	VendorID       uuid.UUID  `db:"vendor_id" json:"-"`
	Name           string     `db:"name" json:"name"`
	CustomerID     *uuid.UUID `db:"customer_id" json:"-"`
	IsAvailable    bool       `db:"is_available" json:"is_available"`
	IsNeedsService bool       `db:"is_needs_service" json:"is_needs_service"`

	Vendor   Vendor    `json:"vendor"` // Use Vendor from vendor.go
	Customer *Customer `json:"customer,omitempty"`
}
