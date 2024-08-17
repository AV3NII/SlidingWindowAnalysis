import pandas as pd


def preprocess_df(df):
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # After some EDA we decided to drop these columns
    cols_to_drop = ['stations', 'description', 'conditions', 'icon']
    df = df.drop(cols_to_drop, axis=1)

    # 'name' is a categorical feature, so we'll one-hot encode it
    df = pd.get_dummies(df, columns=['name'], drop_first=True)

    # Convert 'datetime' column to datetime if it's not already
    if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Set 'datetime' as index if it's not already
    if 'datetime' in df.columns:
        df = df.set_index('datetime')

    # Handle 'sunrise' and 'sunset' columns
    if 'sunrise' in df.columns and 'sunset' in df.columns:
        # Convert to datetime
        df['sunrise'] = pd.to_datetime(df['sunrise'])
        df['sunset'] = pd.to_datetime(df['sunset'])

        # Extract time features
        df['sunrise_hour'] = df['sunrise'].dt.hour + df['sunrise'].dt.minute / 60
        df['sunset_hour'] = df['sunset'].dt.hour + df['sunset'].dt.minute / 60

        # Calculate day length in hours
        df['day_length'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600

        # Drop original 'sunrise' and 'sunset' columns
        df = df.drop(['sunrise', 'sunset'], axis=1)

    return df
