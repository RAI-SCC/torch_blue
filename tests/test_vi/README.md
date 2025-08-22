# Test dependencies

### Complete tests (100% self-coverage)

| directory                      | file                    | target                                 | dependency                                                                                                                         |
|--------------------------------|-------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| test_utils                     | test_metaclass          | utils/post_init_metaclass              | None                                                                                                                               |
|                                | test_init               | utils/init                             | None                                                                                                                               |
|                                | test_use_norm_constants | utils/use_norm_constants               | None                                                                                                                               |
|                                | test_vi_return          | utils/vi_return                        | None                                                                                                                               |
| test_priors                    | test_prior_base         | priors/base                            | test_metaclass                                                                                                                     |
|                                | test_normal_prior       | priors/normal                          | test_prior_base; test_use_norm_constants                                                                                           |
|                                | test_quiet              | priors/quiet                           | test_prior_base; test_use_norm_constants                                                                                           |
| test_variational_distributions | test_vardist_base       | variational_distributions/base         | test_metaclass                                                                                                                     |
|                                | test_normal_vardist     | variational_distributions/normal       | test_vardist_base; test_use_norm_constants                                                                                         |
|                                | test_non_bayesian       | variational_distributions/non_bayesian | test_vardist_base                                                                                                                  |
|                                | test_student_t_vardist  | variational_distributions/student_t    | test_vardist_base; test_use_norm_constants                                                                                         |
| .                              | test_base               | base                                   | test_prior_base; test_vardist_base; test_metaclass; test_vi_return                                                                 |
|                                | test_linear             | linear                                 | test_normal_prior; test_normal_vardist; test_base; test_non_bayesian                                                               |
|                                | test_conv               | conv                                   | test_normal_prior; test_normal_vardist; test_base                                                                                  |
|                                | test_sequential         | sequential                             | test_linear; test_base                                                                                                             |
|                                | test_transformer        | transformer                            | test_base; test_linear; test_normal_prior; test_normal_vardist; test_sequential                                                    |
|                                | test_kl_loss            | kl_loss                                | test_normal_pred; test_predictive_base                                                                                             |
|                                | test_analytical_kl_loss | analytical_kl_loss                     | test_use_norm_constants; test_priors; test_variational_distributions; test_linear; test_sequential; test_kl_loss; test_normal_pred |
| test_predictive_distributions  | test_predictive_base    | base                                   | test_meta                                                                                                                          |
|                                | test_categorical_pred   | predictive_distributions/categorical   | test_predictive_base                                                                                                               |
|                                | test_normal_pred        | predictive_distributions/normal        | test_predicitve_base; test_use_norm_constants                                                                                      |
|                                | test_non_bayesian_pred  | predictive_distributions/non_bayesian  | test_predictive_base                                                                                                               |

### To include tests

| directory | file             | target      | dependency                                                                      |
|-----------|------------------|-------------|---------------------------------------------------------------------------------|
|           |                  |             | ?                                                                               |
