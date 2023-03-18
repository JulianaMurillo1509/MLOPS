from pydantic import BaseModel

class Penguin(BaseModel):
    species : str ="Adelie"
    island : str ="Torgersen"
    bill_length_mm : float = 39.1
    bill_depth_mm : float = 18.7
    flipper_length_mm : float = 181
    body_mass_g : float = 3750
    sex : str ="male"
    year :  int = 2007

