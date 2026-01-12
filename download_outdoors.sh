#!/bin/bash

ENV_FILE=".env"

check_installation() {
  echo "## 🔍 Check Kaggle CLI Installation"
  if command -v kaggle &> /dev/null; then 
    echo " ✅ The 'kaggle' command is installed and accessible."   
    echo "Kaggle version: $(kaggle --version)"
  else
    echo "❌ The 'kaggle' command is NOT installed or not in your PATH."
    echo "Please install it, usually via 'pip install kaggle'."
    exit 1
  fi
}

loading_credentials() {
  echo "---"
  echo "## 📁 Checking for Tokens. You should save that in .env"
  
  if [ -f "$ENV_FILE" ]; then 
    echo "✅ Found and loading variables from '$ENV_FILE'..."
    source "$ENV_FILE"
    echo "$KAGGLE_API_TOKEN"
  else
    echo "⚠️  WARNING: '$ENV_FILE' not found in the current directory."
  fi 

  if [ -n "$KAGGLE_API_TOKEN" ]; then 
    echo "✅ Authentication via KAGGLE_API_TOKEN (from .env or shell)."
    return 0
  fi

  # All checks failed
  echo "❌ No valid authentication credentials found in environment"
  echo "Please ensure '$ENV_FILE' exists and contains 'KAGGLE_API_TOKEN=...'."
  return 1
}


# Function to test authentication (remains the same)
test_authentication() {
    echo "---"
    echo "## 🔑 Testing Authentication"
    
    # The Kaggle CLI automatically finds the file in the standard location.
    # If the file is in the local directory, we rely on the CLI finding it there 
    # (though best practice is to always use the $HOME/.kaggle directory).
    
    # Using a simple, non-resource intensive command to test connectivity
    if kaggle datasets list --sort-by votes --max-size 10240 &> /dev/null; then
        echo "✅ Authentication successful! The Kaggle CLI can connect and fetch data."
        echo "Your Kaggle environment is ready."
    else
        echo "❌ Authentication failed."
        echo "This means the tokens in .env are likely invalid."
        exit 1
    fi
}

# --- Main execution flow ---
check_installation
loading_credentials
test_authentication

echo "---"
echo "## 📥 Starting Dataset Downloads"

ROOT_DATASET="dataset"

# Create the root directory where all data will be stored
echo "Creating root data directory: $ROOT_DATASET"
mkdir -p "$ROOT_DATASET"


# ---

# 1. Indoor Training Set (ITS)
FOLDER_NAME="outdoor-training-set"
TARGET_PATH="$ROOT_DATASET/$FOLDER_NAME"

echo "Downloading: RESIDE STANDARD Beta Outdoor Set (OTS)..."
mkdir -p "$TARGET_PATH"
kaggle datasets download -d brunobelloni/outdoor-training-set-ots-reside \
      --unzip -p "$TARGET_PATH"

echo "✅ All downloads initiated. Check the '$ROOT_DATASET' folder for files."