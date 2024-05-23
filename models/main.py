from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class RefreshToken(Token):
    refresh_token: str


class TokenData(BaseModel):
    sub: str | None = None
    exp: int | None = None
    scopes: list[str]


class UserOauth(BaseModel):
    id: str
    name: str | None = None
    email: str
    image: str | None = None


class AccountOauth(BaseModel):
    userId: str | None = None
    type: str
    provider: str
    providerAccountId: str
    access_token: str
    token_type: str
    expires_at: int | None = None
    id_token: str | None = None
    scope: str
