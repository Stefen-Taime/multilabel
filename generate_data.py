import pandas as pd
import random
from faker import Faker

fake = Faker()

def generate_transport_dataset(num_rows):
    data = []

    for _ in range(num_rows):
        product_id = fake.uuid4()
        net_weight = random.uniform(0.1, 100)
        size = random.choice(['A', 'B', 'C', 'D', 'E'])
        value = random.uniform(50, 10000)
        storage = random.randint(0, 1)
        packaging_cost = random.uniform(1, 50)
        expiry_period = random.randint(1, 365)
        length = random.uniform(1, 100)
        height = random.uniform(1, 100)
        width = random.uniform(1, 100)
        volume = length * height * width
        perishable_index = random.uniform(0, 1)
        flammability_index = random.uniform(0, 1)
        F145 = random.uniform(0, 100)
        F7987 = random.uniform(0, 100)
        F992 = random.uniform(0, 100)
        air = random.randint(0, 1)
        road = random.randint(0, 1)
        rail = random.randint(0, 1)
        sea = random.randint(0, 1)

        row = [product_id, net_weight, size, value, storage, packaging_cost, expiry_period,
               length, height, width, volume, perishable_index, flammability_index,
               F145, F7987, F992, air, road, rail, sea]

        data.append(row)

    columns = ['Product_Id', 'Net_Weight', 'Size', 'Value', 'Storage', 'Packaging_Cost',
               'Expiry_Period', 'Length', 'Height', 'Width', 'Volume', 'Perishable_Index',
               'Flammability_Index', 'F145', 'F7987', 'F992', 'Air', 'Road', 'Rail', 'Sea']

    return pd.DataFrame(data, columns=columns)

# Generate the dataset
num_rows = 2000
transport_dataset = generate_transport_dataset(num_rows)

# Save the dataset to a CSV file
transport_dataset.to_csv("transport_dataset.csv", index=False)
