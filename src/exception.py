import os
import sys


def error_message_details(error,error_details:sys):
    _,_,exe_tb=error_details.exc_info()
    file_name=exe_tb.tb_frame.f_code.co_filename
    message="Error has occured in python script name [{0}] line number[{1}] error message[{2}] ".format(file_name,exe_tb,str(error))
    return message

class CustomException(Exception):
    def __init__(self,error,error_details:sys):
        super().__init__(self,error)
        self.error=error_message_details(error=error,error_details=error_details)