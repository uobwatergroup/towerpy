# Welcome to the Towerpy contributing guide <!-- omit in toc -->

Thank you for investing your time in contributing to our project! :sparkles:.

Read our [Code of Conduct](.github/CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

Use the table of contents icon on the top left corner of this document to get to a specific section of this guide quickly.

## New contributor guide

To get an overview of the project, read the [README](README.md). Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Submitting changes
For small changes to the code or documentation (e.g. correcting a typo), please feel free to open a pull request directly.

For anything more significant, it is really helpful to [open an issue](https://github.com/uobwatergroup/towerpy/issues/new/choose) first, to put it on the project's radar and facilitate discussion.

In the Issues section, it is possible to choose between -Bug report-, -Custom issue template- or -Feature request-, choose the one that feels more apporpiate.

Once a pull request is marked as ready and approved, it will get merged automatically. The automatic merge process will also (try to) ensure the branch is up to date with master, but on occasion you may need to intervene to make sure.

When preparing your branch for a pull request, it would be awesome if you took the time to clean up the history, and in general we prefer that you have kept up to date with master by rebasing as often as necessary.

## Coding style

For Python, we use Spyder's Autopep8 to format the code for conformance to the PEP 8 convention. 
We use the [Numpy conventions](https://numpydoc.readthedocs.io/en/latest/format.html) for our docstrings style.
