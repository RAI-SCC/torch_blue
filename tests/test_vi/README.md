# Test dependencies

### Complete tests (100% self-coverage)

| directory                      | file                    | target                                | dependency                                                                              |
|--------------------------------|-------------------------|---------------------------------------|-----------------------------------------------------------------------------------------|
| test_utils                     | test_metaclass          | utils/post_init_metaclass             | None                                                                                    |
|                                | test_init               | utils/init                            | None                                                                                    |
|                                | test_use_norm_constants | utils/use_norm_constants              | None                                                                                    |
|                                | test_vi_return          | utils/vi_return                       | None                                                                                    |
| test_distributions             | test_distributions_base | distributions/base                    | test_metaclass                                                                          |
|                                | test_categorical        | distributions/categorical             | test_distribution_base                                                                  |
|                                | test_normal             | distributions/normal                  | test_distributions_base; test_use_norm_constants                                        |
|                                | test_non_bayesian       | distributions/non_bayesian            | test_distributions_base                                                                 |
|                                | test_quiet              | distributions/quiet                   | test_distributions_base; test_use_norm_constants                                        |
|                                | test_student_t          | distributions/student_t               | test_distributions_base; test_use_norm_constants                                        |
| .                              | test_base               | base                                  | test_distribution_base; test_metaclass; test_vi_return                                  |
|                                | test_linear             | linear                                | test_normal; test_base; test_non_bayesian                                               |
|                                | test_conv               | conv                                  | test_normal; test_base                                                                  |
|                                | test_sequential         | sequential                            | test_linear; test_base                                                                  |
|                                | test_transformer        | transformer                           | test_base; test_linear; test_normal; test_sequential                                    |
|                                | test_kl_loss            | kl_loss                               | test_normal; test_distributions_base                                                    |
|                                | test_analytical_kl_loss | analytical_kl_loss                    | test_use_norm_constants; test_distributions; test_linear; test_sequential; test_kl_loss |

### To include tests

| directory | file             | target      | dependency                                                                      |
|-----------|------------------|-------------|---------------------------------------------------------------------------------|
|           |                  |             | ?                                                                               |
