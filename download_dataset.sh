#!/bin/bash

ENV_FILE="./.env"

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

echo "Downloading: Indoor Training Set ..."
kaggle datasets download -d balraj98/indoor-training-set-its-residestandard --unzip

# 2. Haze4K
echo "Downloading: Haze4K-T (for training) ..."
kaggle datasets download -d qwertydbooze/haze4k-t --unzip
echo "Downloading: Haze4K-V (for validation) ..."
kaggle datasets download -d qwertydbooze/haze4k-v --unzip

# 3. DenseHaze
echo "Downloading: DenseHaze..."
kaggle datasets download -d sidhantpatel/densehaze --unzip

# 4. O-Haze 
echo "Donwloading: O-Haze ... "
kaggle datasets download -d philiphofmann/o-haze --unzip

echo "✅ All downloads initiated. Files will appear in the current directory."
