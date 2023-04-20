from pydantic import BaseModel

class Covertype(BaseModel):
    Elevation: int =  2991
    Aspect	: int = 119
    Slope	: int = 7
    Horizontal_Distance_To_Hydrology	: int = 67
    Vertical_Distance_To_Hydrology	: int = 11
    Horizontal_Distance_To_Roadways	: int = 1015
    Hillshade_9am	: int = 233
    Hillshade_Noon	: int = 234
    Hillshade_3pm	: int = 133
    Horizontal_Distance_To_Fire_Points	: int = 1570
    Wilderness_Area	: str = "Commanche"
    Soil_Type: str = "C7202"

	
