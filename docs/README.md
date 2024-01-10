COE 379L: Course Materials
===========================

This directory contains the lecture notes for course. They are built with the Sphinx documentation engine.


The best way to work on the materials locally is to use the poetry files to create a virtualenv with the dependencies
installed. You will need to install poetry first; see the documentation: https://python-poetry.org/docs/

With poetry installed, execute the following to install the dependencies in a new virtualenv with:

```
poetry install
```

Once the dependencies have been installed within the new virtualenv, activate the environment:

```
poetry shell
```

Within the activated environment, build the documentation and run the local development server with:

```
make livehtml
```

The docs should be served on http://127.0.0.1:7898/



If locales are not set up on your machine, you may need to set the LC_ALL environment variable:

```
export LC_ALL=C.UTF-8
```
