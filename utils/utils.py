from typing import Annotated
from prisma.models import User
from database.db import db
from models.main import TokenData, UserOauth, AccountOauth
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, ExpiredSignatureError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
import os
from dotenv import load_dotenv
load_dotenv()

oauth2 = OAuth2PasswordBearer(tokenUrl=os.getenv('TOKEN_URL'))
pwd_context = CryptContext(
    schemes=[os.getenv('HASH_ALGORITHM')], deprecated='auto')


async def check_exists(field: str, value: str) -> bool:
    if (await db.user.find_first(where={field: value})):
        return True
    else:
        return False


async def check_exists_account(field: str, value: str) -> bool:
    if (await db.account.find_first(where={field: value})):
        return True
    else:
        return False


async def create_user(username: str, password: str, email: str):
    if (await check_exists('username', username)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already created")
    if (await check_exists('email', email)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already created")
    await db.user.create(data={
        'username': username,
        'password': pwd_context.hash(password),
        'email': email
    })


async def create_user_oauth(user: UserOauth, account: AccountOauth):
    if (await check_exists('email', user.email)):
        # if have email just create account
        if (await check_exists_account('provider', account.provider) and await check_exists_account('providerAccountId', account.providerAccountId)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Account already created")
        else:
            userId = await db.user.find_unique(where={'email': user.email})
            await db.account.create(data={
                'userId': userId.id,
                'type': account.type,
                'provider': account.provider,
                'providerAccountId': account.providerAccountId,
                'access_token': account.access_token,
                'token_type': account.token_type,
                'expires_at': account.expires_at,
                'scope': account.scope
            })
    else:
        await db.user.create(data={
            'id': user.id,
            'email': user.email,
            'emailVerified': datetime.utcnow(),
            'name': user.name,
            'image': user.image,
            'accounts': {
                'create': {
                    'type': account.type,
                    'provider': account.provider,
                    'providerAccountId': account.providerAccountId,
                    'access_token': account.access_token,
                    'token_type': account.token_type,
                    'scope': account.scope
                }
            }
        })


async def get_user(username: str) -> User:

    user = await db.user.find_first(where={'username': username})
    return user


# OK
async def authenticate_user(username: str, password: str) -> User | None:
    user = await get_user(username)
    if user:
        if not pwd_context.verify(password, user.password):
            return None
    return user

# OK


def create_token(data: dict, seconds: int):
    encode_data = data.copy()

    exp = datetime.utcnow()+timedelta(seconds=seconds)
    encode_data['exp'] = exp
    return jwt.encode(encode_data, key=os.getenv('SECRET_KEY'), algorithm=os.getenv('JWT_ALGORITHM'))


async def get_current_user(access_token: Annotated[str, Depends(oauth2)]):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                         detail='Could not validate credentials',
                                         headers={'WWW-Authenticate': 'Bearer'})
    try:
        payload = jwt.decode(access_token, key=os.getenv('SECRET_KEY'),
                             algorithms=[os.getenv('JWT_ALGORITHM')])
        token_data = TokenData(**payload)
    except JWTError as e:
        if isinstance(e, ExpiredSignatureError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            raise credential_exception
    username = token_data.sub
    user = await get_user(username)
    if not user:
        raise credential_exception
    return user


def get_new_access_token(refresh_token: Annotated[str, Depends(oauth2)]):
    try:
        payload = jwt.decode(refresh_token, key=os.getenv('SECRET_KEY'),
                             algorithms=[os.getenv('JWT_ALGORITHM')])

    except JWTError as e:
        if isinstance(e, ExpiredSignatureError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail='Could not validate credentials',
                                headers={'WWW-Authenticate': 'Bearer'})
    access_token = create_token({
        'sub': payload['sub'],
        'scopes': payload['scopes']
    }, seconds=int(os.getenv('ACCESS_TOKEN')))
    return {'access_token': access_token,
            'token_type': 'Bearer'}
