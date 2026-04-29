#!/bin/bash
# Generate a release keystore for signing the APK with v2+v3
# This keystore will be base64 encoded and stored as a GitHub secret

KEYSTORE_DIR="app/keystore"
KEYSTORE_FILE="$KEYSTORE_DIR/release.keystore"
KEYSTORE_PASSWORD="evoai2024"
KEY_ALIAS="evoai"
KEY_PASSWORD="evoai2024"

mkdir -p "$KEYSTORE_DIR"

keytool -genkeypair \
  -v \
  -keystore "$KEYSTORE_FILE" \
  -alias "$KEY_ALIAS" \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000 \
  -storepass "$KEYSTORE_PASSWORD" \
  -keypass "$KEY_PASSWORD" \
  -dname "CN=EvoAI, OU=Development, O=EvoAI Trainer, L=Singapore, ST=Singapore, C=SG"

echo ""
echo "Keystore generated at: $KEYSTORE_FILE"
echo ""
echo "Base64 encoded keystore (for GitHub Secret KEYSTORE_BASE64):"
base64 -w 0 "$KEYSTORE_FILE"
echo ""
echo ""
echo "GitHub Secrets to set:"
echo "  KEYSTORE_BASE64 = <base64 output above>"
echo "  KEYSTORE_PASSWORD = $KEYSTORE_PASSWORD"
echo "  KEY_ALIAS = $KEY_ALIAS"
echo "  KEY_PASSWORD = $KEY_PASSWORD"
