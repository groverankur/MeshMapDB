import hashlib
import os
from typing import Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class AESCipher:
    """
    AES-GCM cipher implementation that uses a per-message salt with Scrypt KDF.

    On encryption:
      - A random 16-byte salt is generated.
      - The provided password (self.password) is derived via Scrypt into a 32-byte AES key.
      - A 12-byte nonce is generated.
      - The output layout is:
            salt (16 bytes) | nonce (12 bytes) | tag (16 bytes) | ciphertext
    On decryption, these fields are extracted accordingly.
    """

    def __init__(self, key: Union[str, bytes], kdf_algo: str = "argon"):
        if isinstance(key, str):
            self.password = key.encode("utf-8")
        else:
            self.password = key

        self.kdf_algo = kdf_algo

    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        kdf_func = {
            "argon": Argon2id(
                salt=salt,
                length=32,
                iterations=1,
                lanes=4,
                memory_cost=64 * 1024,
                ad=None,
                secret=password,
            ),
            "scrypt": Scrypt(
                salt=salt, length=32, n=2**14, r=8, p=1, backend=default_backend()
            ),
        }
        kdf = kdf_func.get(self.kdf_algo, "argon")
        return kdf.derive(password)

    def encrypt(self, data: bytes) -> bytes:
        salt = os.urandom(16)  # New salt for this encryption.
        key = self._derive_key(self.password, salt)
        nonce = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce), backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        tag = encryptor.tag
        checksum = hashlib.sha256(data).digest()
        return salt + nonce + tag + ciphertext + checksum

    def decrypt(self, data: bytes) -> bytes:
        if len(data) < 16 + 12 + 16:
            raise ValueError("Invalid data length for decryption.")
        salt = data[:16]
        nonce = data[16:28]
        tag = data[28:44]
        checksum = data[-32:]
        ciphertext = data[44:-32]
        key = self._derive_key(self.password, salt)
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        if hashlib.sha256(plaintext).digest() != checksum:
            raise ValueError("Data integrity check failed.")
        return plaintext

    def is_encrypted(self, data: bytes) -> bool:
        if len(data) < 16 + 12 + 16 + 32:
            return False
        try:
            salt = data[:16]
            nonce = data[16:28]
            tag = data[28:44]
            checksum = data[-32:]
            ciphertext = data[44:-32]
            if len(ciphertext) <= 0:
                return False
            key = self._derive_key(self.password, salt)
            cipher = Cipher(
                algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()
            )
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return hashlib.sha256(plaintext).digest() == checksum
        except Exception:
            return False
