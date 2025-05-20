from wxwc_converter import get_wxwc_info
from wxwc_converter import create_wxwc_info_file
from updata_infos_for_wxwc import updata_infos_for_wxwc
if __name__ == "__main__":
  #get_wxwc_info("data/wxwc","data/wxwc/train.json")
  create_wxwc_info_file('data/wxwc','data/wxwc','wxwc')
  #updata_infos_for_wxwc('data/wxwc/wxwc_infos_train.pkl','data/wxwc')

