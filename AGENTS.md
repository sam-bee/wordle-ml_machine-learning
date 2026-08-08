# wordle-ml AGENTS.md

You are to assist in creating a machine learning model, using Go and CUDA, to play Wordle. The gomlx library will be
used.

## Version Control

Do not use merge commits. Feature branches going into `master` must be rebased and fast-forward merged. Conflicts with
upstreams are to be resolved by rebase pulls and force-with-lease pushes as appropriate.

Commit messages for smaller commits should resemble the following structure where sensible:

```
To/For/Because/So that [reason for change], [what is changed]
```

For larger commits, use bullet points instead of the above structure where appropriate.

We should, as a general rule, be committing and pushing changes whenever a feature or task has been completed.

In general, the `master` branch should be used for everything. Feature branches should be used only when requested, or
in exceptional cases.

## Systems Administration

In general, agents should not be doing systems administration on the laptop or desktop. Development should generally be
done using docker containers provided, and installing packages etc. within these is acceptable. Agents should under no
circumstances mess around the CUDA drivers on bare metal on development machines.

## Docs

Agents should update documentation as we progress. This goes in `docs/`.

Talk notes can be found in the `talk/` folder. These should be updated too.

## Project Goals

The main goal of the project is to produce a talk for a conference: a 1 hour talk, given to a group of Go developers, on
machine learning. More information is in the `talk/README.md` and `talk/abstract.md` files. Ultimately, it is writing a
good talk that is the priority, not creating a massively complicated machine learning project. Bare this in mind when
discussing the direction of the project.

## Hardware

Both development environments have a 16 core CPU and a 5000-series GPU. The desktop has a 5070 Ti. The laptop has a
5050. (Use of the 5070 Ti is permitted on desktop, however, the project **must not** use the RTX 3060, which may also be
visible.) We are targeting CUDA compute compatibility level 12.0, and no other architectures should be supported. We do
not need code for other devices.

## Programming Style

We never need to run this project on a machine without a CUDA device, so it is completely unnecessary for there to be a
parallel on-host C++ implementation of any CUDA code we generate, even for testing. If it runs on the GPU, it gets
tested there.

It is also important to understand that we are not building a 'flexible' Wordle game. You get 6 turns in a game. The
words have 5 letters. These constants never need to be passed around parametrically, or made configurable through
command line flags, config files, or any other means. We do not write flexible code for things we don't need now. We
keep the code simple instead.
