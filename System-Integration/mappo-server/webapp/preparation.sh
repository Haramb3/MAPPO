echo 'updating the system'
sudo apt-get update

echo 'installing some necessary packages'
sudo apt-get install libpq-dev python-dev

echo 'installing the requirements for the project'
sudo pip3 install -r requirements.txt