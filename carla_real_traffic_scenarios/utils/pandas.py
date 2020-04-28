def swap_columns_inplace(df, c1, c2) -> None:
    TMP_COLUMN_NAME = '___temp'
    df[TMP_COLUMN_NAME] = df[c1]
    df[c1] = df[c2]
    df[c2] = df[TMP_COLUMN_NAME]
    df.drop(columns=[TMP_COLUMN_NAME], inplace=True)