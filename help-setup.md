# Readme

- Application will go on Jetstream

- OpenStack API. For installation on Mac OS X, see [
Installing the Openstack clients on OS X](https://iujetstream.atlassian.net/wiki/spaces/JWT/pages/40796180/Installing+the+Openstack+clients+on+OS+X)

To use,

    source .openrc


### OpenStack Command Line

- See [OpenStack command line](https://iujetstream.atlassian.net/wiki/spaces/JWT/pages/35913730/OpenStack+command+line)
- Followed all of the instructions. To launch an instance, I did

    openstack server create ${OS_USERNAME}-api-U-1 \
        --flavor m1.small \
        --image JS-API-Featured-CentOS7-Intel-Developer-Latest \
        --key-name ${OS_USERNAME}-api-key \
        --security-group ${OS_USERNAME}-global-ssh \
        --nic net-id=${OS_USERNAME}-api-net


- I can ssh in perfectly fine using local SSH keys in this folder.


### Hello, world webserver

- https://www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-centos-7


- Allow traffic into port 80 (http):

    openstack security group rule create --protocol tcp --dst-port 80 --remote-ip 0.0.0.0/0 ${OS_USERNAME}-global-ssh

- Allow traffic into port 443 (https) (not done yet, this will require much more setup)

    openstack security group rule create --protocol tcp --dst-port 443 --remote-ip 0.0.0.0/0 ${OS_USERNAME}-global-ssh

- Allow traffic into port 8501:

    openstack security group rule create --protocol tcp --dst-port 8501 --remote-ip 0.0.0.0/0 ${OS_USERNAME}-global-ssh


- If this is primarily a streamlit application, we need to follow something like the [Streamlit Deployment Guide](https://discuss.streamlit.io/t/streamlit-deployment-guide-wiki/5099).
- Not sure about using `nginx` or `apache` in front of `tornado`.
- Seems like `nginx` is best.

## CentOS7 Latest

- Switched image because the Intel one was completely full (used 20 GB / 20 GB total).
- All works on port 8501! However, still a pain to try to get this working on port 80 with an `nginx` reverse proxy.


Did the possibly questionable:

sudo setsebool httpd_can_network_connect on -P

(see https://serverfault.com/questions/1020429/how-much-does-httpd-can-network-connect-being-set-to-1-actually-open-up-on-selin/1020441#1020441)


    connect() to [::1]:8501 failed (13: Permission denied)


This fixed it (may be questionably secure though)


Annoying to make `nginx` pass things to streamlit ports correctly, but possible!

    streamlit run app.py --server.baseUrlPath=stream


You can go to `http://129.114.17.58/stream/` to see things happen!

This should, in theory, be scalable?!? (Separate streamlit instance for each client / webpage ?!?)

# October 2021

This is all working, sent to `https://js.munano.org/stream`.


# December 2021

- Everything looks good, but I still can't figure out how to make sinatra app work with nginx.
