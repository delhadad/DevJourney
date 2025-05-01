CREATE TABLE user_roles (
    user_id UUID NOT NULL,
    role_id INTEGER NOT NULL,

    PRIMARY KEY (user_id, role_id),

    CONSTRAINT fk_user_id
        FOREIGN KEY (user_id)
          REFERENCES users (id)
          ON DELETE CASCADE,

    CONSTRAINT fk_role_id
        FOREIGN KEY (role_id)
          REFERENCES roles (id)
          ON DELETE CASCADE
);
