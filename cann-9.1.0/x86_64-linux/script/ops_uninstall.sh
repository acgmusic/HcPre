#!/bin/bash

# 请在此处定义各种变量
readonly PACKAGE_SHORT_NAME="ops_A3"
readonly PACKAGE_VERSION_FORM=""
readonly PACKAGE_LOG_NAME="ops_A3"
readonly PACKAGE_ARCH="x86_64"
readonly PYTHON3_INSTALL_CONF="use_private_python.info"
readonly PATH_LENGTH=4096

frame=$(arch)
install_for_all_cmd=""
log_file_name="ascend_${PACKAGE_SHORT_NAME}_install.log" # log文件名字
log_file=""                                              # log文件带路径
if [ "$UID" = "0" ]; then
    LOG_PATH="/var/log/ascend_seclog"
    log_file="${LOG_PATH}/${log_file_name}"
    install_for_all_cmd="--install-for-all"
    PYTHON3_INSTALL_INFO="/etc/${PYTHON3_INSTALL_CONF}"
else
    LOG_PATH="${HOME}/var/log/ascend_seclog"
    log_file="${LOG_PATH}/${log_file_name}"
    PYTHON3_INSTALL_INFO="${HOME}/${PYTHON3_INSTALL_CONF}"
fi

# 路径
script_path="$(dirname $(readlink -f $0))"
form_path="$(cd "$(dirname ${script_path})" && pwd)"
version_path="$(dirname ${form_path})"
install_path="$(dirname ${version_path})"

# 日志模块初始化
function log_init() {
    # 判断输入的安装路径路径是否存在，不存在则创建
    if [ ! -d $LOG_PATH ]; then
        make_dir "$LOG_PATH"
    fi
    if [ ! -f $log_file ]; then
        make_file "$log_file"
        chmod_recursion ${log_file} "640" "log"
        log "INFO" "Log file not found, create a new log file."
    else
        local filesize=$(ls -l $log_file | awk '{ print $5 }')
        local maxsize=$((1024 * 1024 * 50))
        if [ $filesize -gt $maxsize ]; then
            local log_file_move_name="ascend_${PACKAGE_SHORT_NAME}_install_bak.log"
            mv -f ${log_file} ${LOG_PATH}/${log_file_move_name}
            chmod_recursion ${LOG_PATH}/${log_file_move_name} "440" "log"
            make_file "$log_file"
            chmod_recursion ${log_file} "640" "log"
            log "INFO" "log file > 50M, move ${log_file} to ${LOG_PATH}/${log_file_move_name}."
        fi
    fi
    print "INFO" "LogFile:$log_file"
}

# 权限掩码设置
function change_umask() {
    if [ ${UID} -eq 0 ] && [ $(umask) != "0022" ]; then
        print "INFO" "change umask 0022"
        umask 0022
    elif [ ${UID} -ne 0 ] && [ $(umask) != "0002" ]; then
        print "INFO" "change umask 0002"
        umask 0002
    fi
}

# 创建文件夹
function make_dir() {
    change_umask
    print "INFO" "mkdir ${1}"
    mkdir -p ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 fail !"
        exit 1
    fi
}

# 创建文件
function make_file() {
    change_umask
    print "INFO" "touch ${1}"
    touch ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 fail !"
        exit 1
    fi
}

# 递归授权
function chmod_recursion() {
    local parameter2=$2
    local rights="$(echo ${parameter2:0:2})""$(echo ${parameter2:1:1})"
    rights=$([ x${install_for_all_cmd} == x"" ] && echo $2 || echo ${rights})
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -exec chmod ${rights} {} \; 2>/dev/null
        # 日志文件不增加other权限
    elif [ "$3" = "log" ]; then
        find $1 -type f -exec chmod ${parameter2} {} \; 2>/dev/null
    fi
}

# 将日志打印到文件中
function log() {
    if [ x$log_file = x ] || [ ! -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
    elif [ -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2" >>$log_file
    fi
}

# 将关键信息打印到屏幕上
function print() {
    if [ x$log_file = x ] || [ ! -f $log_file ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2"
    else
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +"%Y-%m-%d %H:%M:%S")] [$1]: $2" | tee -a $log_file
    fi
}

# 检查路径字符串
function check_path() {
    local path_str=${1}
    # 判断路径字符串长度
    if [ ${#path_str} -gt ${PATH_LENGTH} ]; then
        print "WARNING" "parameter error $path_str, the length exceeds ${PATH_LENGTH}."
        return 1
    fi
    # 判断是否是绝对路径
    if [[ ! "${path_str}" =~ ^/.* ]]; then
        print "WARNING" "parameter error $path_str, must be an absolute path."
        return 1
    fi
    # 黑名单设置，不允许//，...这样的路径
    if echo "${path_str}" | grep -Eq '\/{2,}|\.{3,}'; then
        print "WARNING" "The path ${path_str} is invalid, cannot contain the following characters: // ...!"
        return 1
    fi
    # 白名单设置，只允许常见字符
    if echo "${path_str}" | grep -Eq '^~?[a-zA-Z0-9./_-]*$'; then
        log "INFO" "The path ${path_str} is correct."
        return 0
    else
        print "WARNING" "The path ${path_str} is invalid, only [a-z,A-Z,0-9,-,_] is support!"
        return 1
    fi
}

# python3环境变量导入
function set_python_environment() {
    if [ -f ${PYTHON3_INSTALL_INFO} ]; then
        local param1=$(cat "${PYTHON3_INSTALL_INFO}" | grep -w "python37_install_path" | cut -d"=" -f2 | sed "s/ //g")
        local param2=$(cat "${PYTHON3_INSTALL_INFO}" | grep -w "python3_install_path" | cut -d"=" -f2 | sed "s/ //g")
        local python3_install_path=$([ x${param1} == x"" ] && echo ${param2} || echo ${param1})
        if [ x"${python3_install_path}" == x"" ]; then
            print "WARNING" "the ${PYTHON3_INSTALL_INFO} file has no python37_install_path or python3_install_path variable. Please check and set it."
        else
            check_path "${python3_install_path}"
            if [ $? -ne 0 ]; then
                print "WARNING" "the ${PYTHON3_INSTALL_INFO} file Python path ${python3_install_path} is error."
            elif [ ! -d "${python3_install_path}" ]; then
                print "WARNING" "the ${PYTHON3_INSTALL_INFO} file Python path ${python3_install_path} is not exist or not directory, please check it."
            else
                if ls ${python3_install_path}/bin/python* 1>/dev/null 2>&1; then
                    export LD_LIBRARY_PATH=${python3_install_path}/lib/:$LD_LIBRARY_PATH
                    export PATH=${python3_install_path}/bin/:$PATH
                    log "INFO" "Setting environment variables ${python3_install_path} succeeded."
                else
                    print "WARNING" "The ${PYTHON3_INSTALL_INFO} file environment variable ${python3_install_path}/bin has no Python binary file, please check and reset it."
                fi
            fi
        fi
    fi
}

# 安全删除文件
function fn_del_file() {
    local file_path=$1
    # 判断变量是否为空
    if [ -n "${file_path}" ]; then
        # 判断是否是文件
        if [ -f "${file_path}" ] || [ -h "${file_path}" ]; then
            rm -f "${file_path}"
            log "INFO" " delete file ${file_path} successfully."
            return 0
        elif ls ${file_path} 1>/dev/null 2>&1; then
            if [ -d ${file_path} ]; then
                log "WARNING" " delete operation, the ${file_path} is directory, not file."
                return 1
            else
                rm -f "${file_path}"
                log "INFO" " delete wildcard file ${file_path} successfully."
                return 0
            fi
        else
            log "WARNING" " delete operation, the file ${file_path} is not exist."
            return 1
        fi
    else
        log "WARNING" " delete operation, file parameter invalid."
        return 2
    fi
}

# 安全删除文件夹
function fn_del_dir() {
    local dir_path=$1
    local is_empty=$2
    # 判断变量不为空且不是系统根盘
    if [ -n "${dir_path}" ] && [[ ! "${dir_path}" =~ ^/+$ ]]; then
        # 判断是否是目录
        if [ -d "${dir_path}" ]; then
            # 判断是否需要判断目录为空不删除
            if [ x"${is_empty}" == x ] || [ "$(ls -A ${dir_path})" = "" ]; then
                chmod -R 700 "${dir_path}" 2>/dev/null
                rm -rf "${dir_path}"
                log "INFO" " delete directory ${dir_path} successfully."
                return 0
            else
                log "WARNING" " delete operation, the directory ${dir_path} is not empty."
                return 1
            fi
        else
            log "ERROR" " delete operation, the ${dir_path} is not exist or not directory."
            return 1
        fi
    else
        log "ERROR" " delete operation, directory parameter invalid."
        return 2
    fi
}

# 失效软链清除和ascend-toolkit文件夹处理
function deal_softlink_and_ascend_toolkit() {
    if [ -d "${version_path}" ]; then
        fn_del_dir "${version_path}" "check_empty"
    fi
    # 删除失效软链接
    find "${install_path}/cann" -type l ! -exec test -e {} \; -delete 2>/dev/null || true
    find "${install_path}/ascend-toolkit/latest" -type l ! -exec test -e {} \; -delete 2>/dev/null || true
    find "${install_path}/ascend-toolkit/set_env.sh" -type l ! -exec test -e {} \; -delete 2>/dev/null || true
    # 删除指定文件夹
    fn_del_dir "${install_path}/ascend-toolkit" "check_empty"
}

function check_result() {
    if [ $? -eq 0 ]; then
        deal_softlink_and_ascend_toolkit
        print "INFO" "${PACKAGE_LOG_NAME} uninstall success"
        exit 0
    else
        print "ERROR" "${PACKAGE_LOG_NAME} uninstall failed, Please refer to the log for more details: ${log_file}"
        exit 1
    fi
}

function deal_ops_uninstall_script() {
    local record_change=n
    if [ ! -w "${form_path}" ]; then
        chmod u+w "${form_path}"
        record_change=y
    fi

    fn_del_file "${form_path}/ascend_ops_install.info"
    fn_del_file "${script_path}/ops_uninstall.sh"
    fn_del_dir "${script_path}" "check_empty"

    if [ -d "${form_path}" ] && [ "${record_change}" = "y" ]; then
        chmod u-w "${form_path}"
    fi
}

function deal_query_pkg_version_script() {
    if [ -f "${version_path}/query_pkg_version.sh" ]; then
        fn_del_file "${version_path}/query_pkg_version.sh"
    fi
}

# 程序开始
function main() {
    # 日志初始化,后续所有模块都有可能使用日志模块必须最先初始化
    log_init
    # python3 环境变量导入
    set_python_environment
    # 图灵总卸载脚本调用
    if [ x"${PACKAGE_VERSION_FORM}" == x"" ] || [ ${frame} == ${PACKAGE_ARCH} ]; then
        if [[ "${PACKAGE_SHORT_NAME}" =~ "toolkit" ]] || [[ "${PACKAGE_SHORT_NAME}" =~ "all" ]]; then
            # Mindstudio软链接移除
            local symlink_file="${version_path}/share/info/mstx/symlink.sh"
            if [ -f $symlink_file ]; then
                bash ${symlink_file} uninstall ${install_path}
            fi
            if [ -f "${version_path}/cann_uninstall.sh" ]; then
                chmod u+w "${version_path}/cann_uninstall.sh"
                sed -i "/uninstall_package \"combo_script\"/d" "${version_path}/cann_uninstall.sh"
                sed -i "/^exit /i uninstall_package \"combo_script\"" "${version_path}/cann_uninstall.sh"
                chmod u-w "${version_path}/cann_uninstall.sh"
                deal_query_pkg_version_script
                ${version_path}/cann_uninstall.sh | tee -a $log_file
                check_result
            else
                print "ERROR" "cann_uninstall.sh file not found"
                exit 1
            fi
        else
            if [ -f "${version_path}/ops_uninstall.sh" ]; then
                deal_ops_uninstall_script
                ${version_path}/ops_uninstall.sh | tee -a $log_file
                check_result
            else
                print "ERROR" "ops_uninstall.sh file not found"
                exit 1
            fi
        fi
    else
        if [ -f "${form_path}/hetero-arch-scripts/cann_uninstall.sh" ]; then
            ${form_path}/hetero-arch-scripts/cann_uninstall.sh | tee -a $log_file
        else
            if [[ "${PACKAGE_SHORT_NAME}" =~ "toolkit" ]] || [[ "${PACKAGE_SHORT_NAME}" =~ "all" ]]; then
                if [ -f "${version_path}/cann_uninstall.sh" ]; then
                    chmod u+w "${version_path}/cann_uninstall.sh"
                    sed -i "/uninstall_package \"combo_script\"/d" "${version_path}/cann_uninstall.sh"
                    sed -i "/^exit /i uninstall_package \"combo_script\"" "${version_path}/cann_uninstall.sh"
                    chmod u-w "${version_path}/cann_uninstall.sh"
                    deal_query_pkg_version_script
                    ${version_path}/cann_uninstall.sh | tee -a $log_file
                    check_result
                else
                    print "ERROR" "cann_uninstall.sh file not found"
                    exit 1
                fi
            else
                if [ -f "${version_path}/ops_uninstall.sh" ]; then
                    deal_ops_uninstall_script
                    ${version_path}/ops_uninstall.sh | tee -a $log_file
                    check_result
                else
                    print "ERROR" "ops_uninstall.sh file not found"
                    exit 1
                fi
            fi
        fi
    fi
}

main $*
