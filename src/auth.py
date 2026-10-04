import hmac


def token_matches(provided_token: str | None, expected_token: str | None) -> bool:
    if not provided_token or not expected_token:
        return False
    return hmac.compare_digest(provided_token, expected_token)


def authenticate_user(
    username: str | None,
    password: str | None,
    user_username: str | None,
    user_password: str | None,
    admin_username: str | None,
    admin_password: str | None,
) -> str | None:
    if token_matches(username, admin_username) and token_matches(password, admin_password):
        return "admin"
    if token_matches(username, user_username) and token_matches(password, user_password):
        return "user"
    return None