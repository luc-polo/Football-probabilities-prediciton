

model = LogisticRegression(fit_intercept=True, random_state = 999, solver = 'liblinear', class_weight = {1:0.6, 0:0.4}, max_iter= 3000, verbose = False, tol=1e-4, n_jobs = -1)