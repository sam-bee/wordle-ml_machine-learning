# wordle-ml AGENTS.md

You are to assist in creating a machine learning model, using Go and CUDA, to play Wordle. The gomlx library will be used.

## Version Control

Do not use merge commits. Feature branches going into `master` must be rebased and fast-forward merged. Conflicts with upstrams are to be resolved by rebase pulls and force-with-lease pushes as appropriate.

Commit messages for smaller commits should resemble the following structure where sensible:

```
To/For/Because/So that [reason for change], [what is changed]
```

For larger commits, use bullet points instead of the above structure where appropriate.

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
