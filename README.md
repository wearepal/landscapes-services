# Landscapes Services 

Accessible via: https://landscapes.wearepal.ai/api/

## Deployment

* Ensure a network is available named ``landscapes_services``, this is needed to connect to the Landscape Modelling Tool
* Clone this git repository onto the server, and cd into it
* Run ``docker stop landscapes-services`` then ``docker rm landscapes-services`` to remove old version of the application
* Build the application with ``docker build --platform "linux/amd64" -t landscapes-services .``
* Deploy with ``docker run -d --name landscapes-services --network landscapes_services -p 5001:5001 landscapes-services``

## Entry Points

### v1/segment (get)

TODO: Write up documentation for Segment API and it's parameters.
