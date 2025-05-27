# Create a virtual envoriment called 'virt_env'
python -m venv virt_env

# Activate the virtual envoriment
source ./virt_env/bin/activate

# Install requirements
#pip install --upgrade pip
#pip install -r requirements.txt

# Download data from Hugging Face
python src/read_data.py "$@"

# Inform user
echo "Successfully installed requirements and downdloaded the data from Hugging Face"