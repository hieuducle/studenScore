import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from lazypredict.Supervised import LazyRegressor



data = pd.read_csv("StudentScore.xls", delimiter=",")
# print(data.corr())
target = "math score"

# sn.histplot(data["math score"])
# plt.title("Math score distribution")
# plt.savefig("MathDistribution.png")

x = data.drop(target, axis=1)
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])
nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OneHotEncoder())
])

# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train["race/ethnicity"], result):
#     print("Before {}. After {}".format(i,j))
preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ordinal_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_features", nom_transformer, ["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor()),
])

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

# metrics
# print("MAE {}".format(mean_absolute_error(y_test, y_pred)))
# print("MSE {}".format(mean_squared_error(y_test, y_pred)))
# print("R2 {}".format(r2_score(y_test, y_pred)))


# for i, j in zip(y_pred, y_test):
#     print("y_pred {} - y_test {}".format(i, j))


# x_train = reg.fit_transform(x_train)
# x_test = reg.transform(x_test)

# lazy_predict
# lazy_reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None )
# models, predictions = lazy_reg.fit(x_train, x_test, y_train, y_test)
# print(predictions)


# parameters = {
#     "regressor__n_estimators": [50, 100, 200, 500],
#     "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
#     "regressor__max_depth": [None, 5, 10, 20],
#     "regressor__max_features": ["sqrt", "log2"],
#     "preprocessor__num_features__imputer__strategy": ["mean", "median"],
# }
# # model = GridSearchCV(reg, param_grid=parameters, scoring="r2", cv=6, verbose=2, n_jobs=8)
# model = RandomizedSearchCV(reg, param_distributions=parameters, scoring="r2", cv=6, verbose=1, n_iter=20, n_jobs=8)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)