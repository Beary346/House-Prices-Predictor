from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def build_pipeline():
    # All Features
    numeric_features = ["MSSubClass", 
                        "LotFrontage",
                        "LotArea",
                        "OverallQual",
                        "OverallCond",
                        "YearBuilt",
                        "YearRemodAdd",
                        "MasVnrArea",
                        "BsmtFinSF1",
                        "BsmtFinSF2",
                        "BsmtUnfSF",
                        "TotalBsmtSF",
                        "1stFlrSF",
                        "2ndFlrSF",
                        "LowQualFinSF",
                        "GrLivArea",
                        "BsmtFullBath",
                        "BsmtHalfBath",
                        "FullBath",
                        "HalfBath",
                        "BedroomAbvGr",
                        "KitchenAbvGr",
                        "TotRmsAbvGrd",
                        "Fireplaces",
                        "GarageYrBlt",
                        "GarageCars",
                        "GarageArea",
                        "WoodDeckSF",
                        "OpenPorchSF",
                        "EnclosedPorch",
                        "3SsnPorch",
                        "ScreenPorch",
                        "PoolArea",
                        "MiscVal",
                        "MoSold",
                        "YrSold",
                        ]
    catagoric_features = ["MSZoning",
                          "Street",
                          "Alley",
                          "LotShape",
                          "LandContour",
                          "Utilities",
                          "LotConfig",
                          "LandSlope",
                          "Neighborhood",
                          "Condition1",
                          "Condition2",
                          "BldgType",
                          "HouseStyle",
                          "RoofStyle",
                          "RoofMatl",
                          "Exterior1st",
                          "Exterior2nd",
                          "MasVnrType",
                          "ExterQual",
                          "ExterCond",
                          "Foundation",
                          "BsmtQual",
                          "BsmtCond",
                          "BsmtExposure",
                          "BsmtFinType1",
                          "BsmtFinType2",
                          "Heating",
                          "HeatingQC",
                          "CentralAir",
                          "Electrical",
                          "KitchenQual",
                          "Functional",
                          "FireplaceQu",
                          "GarageType",
                          "GarageFinish",
                          "GarageQual",
                          "GarageCond",
                          "PavedDrive",
                          "PoolQC",
                          "Fence",
                          "MiscFeature",
                          "SaleType",
                          "SaleCondition"]
    
    numeric_transformer = Pipeline([
        ("inputer", SimpleImputer(strategy="median")), # Could Change Score If Tweaked
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), catagoric_features)
        ]
    )

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestRegressor(n_estimators=500, max_depth=10, criterion='squared_error'))
    ])