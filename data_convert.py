import os
from Evtx.Evtx import FileHeader
import contextlib
import mmap
from Evtx.Views import evtx_file_xml_view
import pandas as pd
from xml.dom import minidom


def handle_folder(data_path):
    print(f"current path: {data_path}")
    files = os.listdir(data_path)
    for data in files:
        if os.path.isdir(data_path + data):
            handle_folder(data_path + data + '\\')
        else:
            print(f"current file: {data_path + data}")
            if data[len(data) - 5:] == '.evtx':
                read_evtx(data_path, data)


def get_system_data(xml_tree, tag):
    tag_data = xml_tree.getElementsByTagName(tag)
    if len(tag_data) == 0:
        return None
    else:
        tag_data = tag_data[0].childNodes
        if len(tag_data) == 0:
            return None
        else:
            return tag_data[0].data


def get_system_attr(xml_tree, tag, attr):
    tag_data = xml_tree.getElementsByTagName(tag)
    if len(tag_data) == 0:
        return None
    else:
        return tag_data[0].getAttribute(attr)


def read_evtx(filepath, filename):
    # system_tag_list = ['EventID', 'Version', 'Level', 'Task', 'Opcode', 'Keywords', 'EventRecordID', 'Channel',
    # 'Computer']
    with open(filepath + filename, 'r') as file:
        with contextlib.closing(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)) as logs:
            counter = 0
            header = FileHeader(logs, 0)
            whole_evtx_dict = {}
            try:
                for xml, records in evtx_file_xml_view(header):
                    whole_content_dict = {}
                    event_data_dict = {}
                    xml_tree = minidom.parseString(xml)
                    guid = get_system_attr(xml_tree, 'Provider', 'Guid')
                    whole_content_dict['Name'] = get_system_attr(xml_tree, 'Provider', 'Name')
                    whole_content_dict['Guid'] = guid
                    whole_content_dict['EventID'] = get_system_data(xml_tree, 'EventID')
                    whole_content_dict['Version'] = get_system_data(xml_tree, 'Version')
                    whole_content_dict['Level'] = get_system_data(xml_tree, 'Level')
                    whole_content_dict['Task'] = get_system_data(xml_tree, 'Task')
                    whole_content_dict['Opcode'] = get_system_data(xml_tree, 'Opcode')
                    whole_content_dict['Keywords'] = get_system_data(xml_tree, 'Keywords')
                    whole_content_dict['EventRecordID'] = get_system_data(xml_tree, 'EventRecordID')
                    whole_content_dict['Channel'] = get_system_data(xml_tree, 'Channel')
                    whole_content_dict['Computer'] = get_system_data(xml_tree, 'Computer')
                    whole_content_dict['ActivityID'] = get_system_attr(xml_tree, 'Correlation', 'ActivityID')
                    whole_content_dict['ProcessID'] = get_system_attr(xml_tree, 'Execution', 'ProcessID')
                    whole_content_dict['ThreadID'] = get_system_attr(xml_tree, 'Execution', 'ThreadID')
                    whole_content_dict['SystemTime'] = get_system_attr(xml_tree, 'TimeCreated', 'SystemTime')
                    for data_record in xml_tree.getElementsByTagName('Data'):
                        sub_nodes = data_record.childNodes
                        if len(sub_nodes) == 0:
                            event_data_dict[data_record.getAttribute('Name')] = None
                        else:
                            for node in sub_nodes:
                                event_data_dict[data_record.getAttribute('Name')] = node.data
                    whole_content_dict['EventData'] = event_data_dict
                    whole_evtx_dict[counter] = whole_content_dict
                    counter += 1
                pass
            except Exception as e:
                print(f"Caught an unexpected error: {e}")
                pass
    result_df = pd.DataFrame(whole_evtx_dict)
    result_df.to_csv(filepath + filename[: len(filename) - 5] + '.csv', index=True)


handle_folder('E:\\Codes\\PythonProjects\\Windows-Security-Event-Log-Analysis\\Errors\\')
# result_df = pd.DataFrame(result_dict)
# result_df.to_csv('D:\\Code\\pythonProject\\evtv.csv', index=True)
