import pandas as pd
import logging
from sklearn.cluster import KMeans

def intelligent_sample_anomalies(df: pd.DataFrame, target_ratio: float = 0.15) -> pd.DataFrame:
    """
    Intelligently reduce the number of anomaly sessions to reach a target ratio
    using K-Means clustering.
    """
    normal_df = df[df['is_anomaly'] == 0]
    attack_df = df[df['is_anomaly'] == 1]
    
    feature_cols = [
        'request_count', 'error_rate', 'auth_failure_count', 
        'avg_response_time_ms', 'p95_response_time_ms', 'unique_endpoints',
        'unique_ips', 'anomalous_path_count', 'post_ratio', 
        'js_error_count', 'request_rate', 'session_duration_s'
    ]
    
    attack_unique = attack_df.drop_duplicates(subset=feature_cols)
    logging.info(f"  Deduplication: Reduced {len(attack_df)} raw attacks to {len(attack_unique)} unique patterns.")
    
    n_normal = len(normal_df)
    n_attack_target = int((target_ratio * n_normal) / (1 - target_ratio))
    
    if len(attack_unique) > n_attack_target and n_attack_target > 0:
        logging.info(f"  Clustering: Picking {n_attack_target} representative samples using K-Means...")
        kmeans = KMeans(n_clusters=n_attack_target, random_state=42, n_init='auto')
        attack_unique = attack_unique.copy()
        attack_unique['cluster'] = kmeans.fit_predict(attack_unique[feature_cols])
        
        attack_final = attack_unique.groupby('cluster').first().reset_index(drop=True)
    else:
        attack_final = attack_unique
        
    if len(attack_final) < n_attack_target:
        diff = n_attack_target - len(attack_final)
        remaining = attack_df[~attack_df.index.isin(attack_final.index)]
        if not remaining.empty:
            extra = remaining.sample(n=min(len(remaining), diff), random_state=42)
            attack_final = pd.concat([attack_final, extra])

    balanced_df = pd.concat([normal_df, attack_final]).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df
