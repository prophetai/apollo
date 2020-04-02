![tests](https://github.com/prophetai/apollo/workflows/tests/badge.svg)
# Apollo
Apollo is a service that delivers predictions on forex markets to its user basis.

It's designed to work as a cronjob in a Kubernetes cluster in [Google Cloud Platform ](http://cloud.google.com)(but it should work anywhere else too).

## Are there any requirements to use it?
Yes, just a few:

- Python >=3.6
- [Docker](https://docs.docker.com/docker-for-mac/install/#install-and-run-docker-for-mac)

You need also two important things:
  - The add the `/models` folder inside `/apollo` with the models(.h5) and variable names (.csv) that create the predictions
  - Filling in an `.env` file with the needed environment variables

You can find an example of the `.env` file named`.env.dist` at `apollo/src`


## How do I deploy it?
If `.env` is filled correctly. Run this couple of commands to start:

This should build the docker container ready to run
```
$ make build
```
And this should run the built container on you local machine.
```
$ make run
```
To check if everything ran correctly you can check your running container logs:
```
$ docker logs -f [DOCKER CONTAINER ID]
```
## How can I deploy it to the cloud?
Ok, here are the instructions to make it work in GCP:

First you need somewhere to run your container. Lets do a Kubernetes cluster:

```
gcloud container clusters create [CLUSTER_NAME] [--zone [COMPUTE_ZONE]]
```

After that you need to upload your container somewhere in order for your Kubernetes pods to run it.

If you want to upload it to GCP. First check you are logged in the GCP CLI and then use this:
```
$ make push
```
After that you can make a cronjob in your cluster like this:

```
$ make cronjob
```
This should make a cronjob that runs the service every hour. If you want to change the cadence of the running you can check the `cronjob.yaml` file.
