from fastapi import APIRouter, HTTPException, status, Depends, Security, Form, Body
from fastapi.security import OAuth2PasswordRequestForm
from prisma.models import User
from models.main import Token, RefreshToken, UserOauth, AccountOauth
from typing import Annotated
from utils.utils import create_user, authenticate_user, create_token, get_current_user, get_new_access_token, create_user_oauth
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter(
    prefix='/user',
    tags=['user']
)


@router.post('/signup')
async def sign_up(username: Annotated[str, Form()], password: Annotated[str, Form()], email: Annotated[str, Form()]):
    await create_user(username, password, email)
    return status.HTTP_200_OK


@router.post('/signup-oauth')
async def signup_oauth(user: UserOauth, account: AccountOauth):
    await create_user_oauth(user, account)
    return status.HTTP_200_OK


@router.post('/token', response_model=RefreshToken)
async def login_for_token(login_form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user(login_form.username, login_form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Incorrect username or password')
    access_token = create_token(
        {'sub': login_form.username,
         'scopes': login_form.scopes},
        seconds=int(os.getenv('ACCESS_TOKEN'))
    )
    refresh_token = create_token(
        {'sub': login_form.username,
         'scopes': login_form.scopes},
        seconds=int(os.getenv('REFRESH_TOKEN'))
    )
    return {'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'bearer'}


@router.get('/refresh-token', response_model=Token)
async def refresh_token(token: Annotated[dict, Depends(get_new_access_token)]):
    return token


@router.get('', response_model=User)
async def read_user_info(user: Annotated[User, Depends(get_current_user)]):
    return user
