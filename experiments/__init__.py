import os

# To avoid openMP issue on macos, set flag below as workaround to single OpenMP runtime constraint
# (note that this may cause crashes or silently produce incorrect results)
# https://github.com/dmlc/xgboost/issues/1715
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
