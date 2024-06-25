# This script pulls any changes from MCCS station YAML
GITDIR=ska-low-deployment

if [ -f $GITDIR ]; then
    cd $GITDIR;
    git pull origin main;
else
   git clone https://gitlab.com/ska-telescope/ska-low-deployment;
fi
