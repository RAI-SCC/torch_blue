## How to contribute to torch_bayesian


#### **Did you find a bug?**

* **Ensure the bug was not already reported** by searching on GitHub under
[Issues](https://github.com/RAI-SCC/torchbuq/issues).

* If you're unable to find an open issue addressing the problem,
[open a new one](https://github.com/RAI-SCC/torchbuq/issues/new). Be sure to include a
**title and clear description**, as much relevant information as possible. Ideally a
**code sample** or a **pytest test** demonstrating the expected behavior that is not
occurring.

#### **Did you write a patch that fixes a bug?**

* Awesome, thanks for helping.

* Ensure your code follows the [coding conventions](#coding-conventions).

* Open a new GitHub pull request into the `dev` branch with the patch.

* Ensure the PR description clearly describes the problem and solution. Include the
relevant issue number if applicable.

#### **Did you implement a new distribution or layer type?**

* Awesome, thanks for growing the zoo.

* Ensure your code follows the [coding conventions](#coding-conventions).

* Open a new GitHub pull request into the `dev` branch with the patch.

#### **Do you intend to add a new feature or change an existing one?**

* Open an issue about the planned change early so we can give feedback on the viability
of your idea.

### Coding conventions

 * torch_bayesian uses mypy and ruff to enforce type hinting and coding style.
Installing the dev dependencies and running pre-commit will check (and possibly fix)
the code style for you.

* We are currently maintaining 100% test coverage. Try and do your part to keep this up :smile:

* Check  the [tests README](https://github.com/RAI-SCC/torchbuq/blob/main/tests/test_vi/README.md).
Every file has an associated test which is supposed to reach full coverage on that
specific file. The README also tracks the loop free dependency tree of the tests. Please
do not add loops.
