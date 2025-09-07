from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

SECRET_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAislAa9XngWX6Hpx+9KJo
xyrxVKPO1eSv2KEvIGhzQ7jWTlX50wDLbY3u/6mKTCuyrXLTSfNubpYWRbBwlBJC
oi56Ccn5259XipzdcisI3HyYMWgfOJxvNQN40CnnAJxn/wn0b6YsvWutMWUjMP3G
1h3WMBz4KPmS/iu9Zkg+dUc7NCYO54d5leKt51Iu1iEBJ0x6BCCi16052lPqadxO
ZPVbkgVH4zoMFEL7HV1apOiK3AiGP57BwAMB81xLZyvfdKmnWuVZpTXCbzG1mtYu
Xj6GuRPpscYqlv+xilOO/Q2BEXpauR1OCleojxldpK6GusApqzVxwY0K7E5qGDhg
VQIDAQAB
-----END PUBLIC KEY-----"""
ALGORITHM = "RS256"

bearer_scheme = HTTPBearer()

def validate_jwt_token(credentials: HTTPAuthorizationCredentials):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
