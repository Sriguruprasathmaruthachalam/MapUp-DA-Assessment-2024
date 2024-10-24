from typing import Dict, List

import pandas as pd



def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    
    for i in range(0, len(lst), n):        
        group = lst[i:i+n]        
       
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        
        result.extend(reversed_group)
        
    return lst

    


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here   

    length_dict = {}    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)    
    
    return dict(sorted(length_dict.items()))



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    from typing import Any, Dict, List

    def flatten(current_dict: Dict, parent_key: str = '') -> Dict:
        items = {}
        
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.update(flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(flatten(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v
        
        return items
    
    return flatten(nested_dict)




   


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    from typing import List

    def backtrack(start: int):
        if start == len(nums):
            # Add a copy of the current permutation to the result
            result.append(nums[:])
            return
        
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            # Swap the current index with the start index
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)  # Recur with the next index
            # Backtrack (swap back)
            nums[start], nums[i] = nums[i], nums[start]
    
    result = []
    nums.sort()  # Sort to handle duplicates
    backtrack(0)
    return result





def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
        
    patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b',  
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])-(\d{4})\b',    
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b'   
    ]

    valid_dates = []
    
    for pattern in patterns:
        found_dates = re.findall(pattern, text)
        for date in found_dates:
            if pattern == patterns[0]:  
                valid_dates.append(f"{date[0]}-{date[1]}-{date[2]}")
            elif pattern == patterns[1]:  
                valid_dates.append(f"{date[0]}/{date[1]}-{date[2]}")
            elif pattern == patterns[2]: 
                valid_dates.append(f"{date[0]}.{date[1]}.{date[2]}")
    
    return valid_dates




def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """  

  
    coordinates = polyline.decode(polyline_str)
    
        df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
       distances = [0.0]  
    for i in range(1, len(df)):
        dist = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(dist)
    
    df['distance'] = distances
    return df



 


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    from typing import List   
    
    rotated = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]    
    
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):            
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            result[i][j] = row_sum + col_sum
            
    return result




def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
  
    grouped = df.groupby(['id', 'id_2'])

    def check_completeness(group):
        
        full_days = pd.date_range(start=group['start_timestamp'].min().normalize(), 
                                   end=group['end_timestamp'].max().normalize() + pd.Timedelta(days=6), 
                                   freq='D')

       
        all_days_covered = all(day in group['start_timestamp'].dt.date.unique() for day in full_days.date)
        full_24_hour_coverage = (group['start_timestamp'].min().time() <= pd.Timestamp('00:00:00').time() and 
                                  group['end_timestamp'].max().time() >= pd.Timestamp('23:59:59').time())
        
        return not (all_days_covered and full_24_hour_coverage)

    
    result = grouped.apply(check_completeness)

  
    return result


   
