# Get unrar in case our system does not have it
apt update
apt install unrar

# First, download annotation data
wget https://tubic.org/doric/static/doric_data/doric12.1.rar

# Extract data
unrar x doric12.1.rar

# Download sequences