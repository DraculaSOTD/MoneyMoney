import os
import json
import base64
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path
import keyring
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class KeyProvider(Enum):
    ENVIRONMENT = "environment"
    FILE = "file"
    KEYRING = "keyring"
    AWS_SECRETS = "aws_secrets"
    HASHICORP_VAULT = "hashicorp_vault"

@dataclass
class APIKey:
    name: str
    key: str
    secret: Optional[str] = None
    exchange: Optional[str] = None
    permissions: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class SecureKeyManager:
    def __init__(self, provider: KeyProvider = KeyProvider.ENVIRONMENT, 
                 master_password: Optional[str] = None,
                 key_file_path: Optional[str] = None):
        self.provider = provider
        self.key_file_path = key_file_path or Path.home() / ".crypto_trading" / "keys.enc"
        self._fernet: Optional[Fernet] = None
        self._keys: Dict[str, APIKey] = {}
        
        if provider in [KeyProvider.FILE, KeyProvider.KEYRING]:
            if not master_password:
                master_password = os.getenv("MASTER_PASSWORD", "")
            if not master_password:
                raise ValueError("Master password required for encrypted storage")
            self._init_encryption(master_password)
            
        self._load_keys()
        
    def _init_encryption(self, master_password: str):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt_for_key_derivation',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self._fernet = Fernet(key)
        
    def _load_keys(self):
        try:
            if self.provider == KeyProvider.ENVIRONMENT:
                self._load_from_environment()
            elif self.provider == KeyProvider.FILE:
                self._load_from_file()
            elif self.provider == KeyProvider.KEYRING:
                self._load_from_keyring()
            elif self.provider == KeyProvider.AWS_SECRETS:
                self._load_from_aws()
            elif self.provider == KeyProvider.HASHICORP_VAULT:
                self._load_from_vault()
                
            logger.info(f"Loaded {len(self._keys)} API keys from {self.provider.value}")
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            
    def _load_from_environment(self):
        # Binance keys
        if os.getenv("BINANCE_API_KEY"):
            self._keys["binance"] = APIKey(
                name="binance",
                key=os.getenv("BINANCE_API_KEY"),
                secret=os.getenv("BINANCE_API_SECRET"),
                exchange="binance",
                permissions={"trading": True, "withdrawal": False}
            )
            
        # Coinbase keys
        if os.getenv("COINBASE_API_KEY"):
            self._keys["coinbase"] = APIKey(
                name="coinbase",
                key=os.getenv("COINBASE_API_KEY"),
                secret=os.getenv("COINBASE_API_SECRET"),
                exchange="coinbase",
                permissions={"trading": True, "withdrawal": False}
            )
            
        # Alternative data source keys
        if os.getenv("TWITTER_API_KEY"):
            self._keys["twitter"] = APIKey(
                name="twitter",
                key=os.getenv("TWITTER_API_KEY"),
                secret=os.getenv("TWITTER_API_SECRET"),
                metadata={"bearer_token": os.getenv("TWITTER_BEARER_TOKEN")}
            )
            
        if os.getenv("NEWS_API_KEY"):
            self._keys["newsapi"] = APIKey(
                name="newsapi",
                key=os.getenv("NEWS_API_KEY")
            )
            
    def _load_from_file(self):
        if not self.key_file_path.exists():
            logger.warning(f"Key file not found: {self.key_file_path}")
            return
            
        try:
            with open(self.key_file_path, 'rb') as f:
                encrypted_data = f.read()
                
            decrypted_data = self._fernet.decrypt(encrypted_data)
            keys_data = json.loads(decrypted_data.decode())
            
            for key_name, key_dict in keys_data.items():
                self._keys[key_name] = APIKey(**key_dict)
                
        except Exception as e:
            logger.error(f"Failed to load keys from file: {e}")
            
    def _save_to_file(self):
        if self.provider != KeyProvider.FILE:
            return
            
        try:
            self.key_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            keys_data = {
                name: asdict(key) for name, key in self._keys.items()
            }
            
            # Convert datetime objects to strings
            for key_data in keys_data.values():
                if key_data.get('created_at'):
                    key_data['created_at'] = key_data['created_at'].isoformat()
                if key_data.get('expires_at'):
                    key_data['expires_at'] = key_data['expires_at'].isoformat()
                    
            json_data = json.dumps(keys_data).encode()
            encrypted_data = self._fernet.encrypt(json_data)
            
            with open(self.key_file_path, 'wb') as f:
                f.write(encrypted_data)
                
            os.chmod(self.key_file_path, 0o600)  # Read/write for owner only
            logger.info(f"Saved {len(self._keys)} keys to encrypted file")
            
        except Exception as e:
            logger.error(f"Failed to save keys to file: {e}")
            
    def _load_from_keyring(self):
        try:
            service_name = "crypto_trading_bot"
            
            # Get list of stored keys
            stored_keys_json = keyring.get_password(service_name, "key_list")
            if not stored_keys_json:
                return
                
            key_names = json.loads(stored_keys_json)
            
            for key_name in key_names:
                key_data_json = keyring.get_password(service_name, key_name)
                if key_data_json:
                    key_dict = json.loads(key_data_json)
                    self._keys[key_name] = APIKey(**key_dict)
                    
        except Exception as e:
            logger.error(f"Failed to load from keyring: {e}")
            
    def _save_to_keyring(self):
        if self.provider != KeyProvider.KEYRING:
            return
            
        try:
            service_name = "crypto_trading_bot"
            
            # Save list of key names
            key_names = list(self._keys.keys())
            keyring.set_password(service_name, "key_list", json.dumps(key_names))
            
            # Save each key
            for key_name, api_key in self._keys.items():
                key_dict = asdict(api_key)
                
                # Convert datetime objects
                if key_dict.get('created_at'):
                    key_dict['created_at'] = key_dict['created_at'].isoformat()
                if key_dict.get('expires_at'):
                    key_dict['expires_at'] = key_dict['expires_at'].isoformat()
                    
                keyring.set_password(service_name, key_name, json.dumps(key_dict))
                
            logger.info(f"Saved {len(self._keys)} keys to system keyring")
            
        except Exception as e:
            logger.error(f"Failed to save to keyring: {e}")
            
    def _load_from_aws(self):
        try:
            import boto3
            client = boto3.client('secretsmanager')
            
            # List all secrets with specific tag
            response = client.list_secrets(
                Filters=[{'Key': 'tag-key', 'Values': ['crypto-trading']}]
            )
            
            for secret in response['SecretList']:
                secret_value = client.get_secret_value(SecretId=secret['Name'])
                key_data = json.loads(secret_value['SecretString'])
                
                self._keys[secret['Name']] = APIKey(**key_data)
                
        except ImportError:
            logger.error("boto3 not installed for AWS Secrets Manager")
        except Exception as e:
            logger.error(f"Failed to load from AWS: {e}")
            
    def _load_from_vault(self):
        try:
            import hvac
            
            vault_url = os.getenv("VAULT_URL", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")
            
            if not vault_token:
                logger.error("VAULT_TOKEN not set")
                return
                
            client = hvac.Client(url=vault_url, token=vault_token)
            
            if not client.is_authenticated():
                logger.error("Failed to authenticate with Vault")
                return
                
            # Read keys from Vault KV v2 engine
            mount_point = "secret"
            path = "crypto-trading/keys"
            
            response = client.secrets.kv.v2.read_secret_version(
                mount_point=mount_point,
                path=path
            )
            
            for key_name, key_data in response['data']['data'].items():
                self._keys[key_name] = APIKey(**key_data)
                
        except ImportError:
            logger.error("hvac not installed for HashiCorp Vault")
        except Exception as e:
            logger.error(f"Failed to load from Vault: {e}")
            
    def add_key(self, api_key: APIKey):
        if not api_key.created_at:
            api_key.created_at = datetime.now()
            
        self._keys[api_key.name] = api_key
        
        if self.provider == KeyProvider.FILE:
            self._save_to_file()
        elif self.provider == KeyProvider.KEYRING:
            self._save_to_keyring()
            
        logger.info(f"Added API key: {api_key.name}")
        
    def get_key(self, name: str) -> Optional[APIKey]:
        key = self._keys.get(name)
        
        if key and key.expires_at and key.expires_at < datetime.now():
            logger.warning(f"API key {name} has expired")
            return None
            
        return key
        
    def get_exchange_keys(self, exchange: str) -> Optional[APIKey]:
        for key in self._keys.values():
            if key.exchange == exchange:
                return key
        return None
        
    def remove_key(self, name: str) -> bool:
        if name in self._keys:
            del self._keys[name]
            
            if self.provider == KeyProvider.FILE:
                self._save_to_file()
            elif self.provider == KeyProvider.KEYRING:
                self._save_to_keyring()
                
            logger.info(f"Removed API key: {name}")
            return True
            
        return False
        
    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "exchange": key.exchange,
                "permissions": key.permissions,
                "created_at": key.created_at.isoformat() if key.created_at else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "is_expired": key.expires_at < datetime.now() if key.expires_at else False
            }
            for name, key in self._keys.items()
        }
        
    def rotate_key(self, name: str, new_key: str, new_secret: Optional[str] = None) -> bool:
        if name not in self._keys:
            logger.error(f"Key {name} not found")
            return False
            
        old_key = self._keys[name]
        
        # Create new key with same settings
        new_api_key = APIKey(
            name=name,
            key=new_key,
            secret=new_secret,
            exchange=old_key.exchange,
            permissions=old_key.permissions,
            created_at=datetime.now(),
            expires_at=old_key.expires_at,
            metadata=old_key.metadata
        )
        
        # Archive old key
        if old_key.metadata is None:
            old_key.metadata = {}
        old_key.metadata['rotated_at'] = datetime.now().isoformat()
        old_key.metadata['replaced_by'] = new_key[:8] + "..."
        
        # Save archived key with timestamp
        archive_name = f"{name}_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._keys[archive_name] = old_key
        
        # Replace with new key
        self._keys[name] = new_api_key
        
        if self.provider == KeyProvider.FILE:
            self._save_to_file()
        elif self.provider == KeyProvider.KEYRING:
            self._save_to_keyring()
            
        logger.info(f"Rotated API key: {name}")
        return True
        
    def export_public_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "keys_count": len(self._keys),
            "exchanges": list(set(k.exchange for k in self._keys.values() if k.exchange)),
            "has_twitter": "twitter" in self._keys,
            "has_news": "newsapi" in self._keys
        }