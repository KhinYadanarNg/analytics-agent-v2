from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get JWT secret key from environment and handle newline escapes
jwt_key_from_env = os.getenv("JWT_SECRET_KEY")
jwt_key_from_env =jwt_key_from_env.replace('\\n', '\n')

ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")

bearer_scheme = HTTPBearer()

def validate_jwt_token(credentials: HTTPAuthorizationCredentials):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, jwt_key_from_env, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
