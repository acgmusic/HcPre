//===----------------------------------------------------------------------===//
//
// Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ASRTC_ASRTC_H
#define LLVM_CLANG_ASRTC_ASRTC_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a program object.
typedef void *asrtcProgram;

/// Error codes returned by ASRTC APIs.
typedef enum {
  ASRTC_SUCCESS = 0,
  ASRTC_ERROR_OUT_OF_MEMORY,
  ASRTC_ERROR_PROGRAM_CREATION_FAILURE,
  ASRTC_ERROR_INVALID_PROGRAM,
  ASRTC_ERROR_INVALID_INPUT,
  ASRTC_ERROR_INVALID_OPTION,
  ASRTC_ERROR_COMPILE,
  ASRTC_ERROR_LINK,
  ASRTC_ERROR_NOT_IMPLEMENTED,
  ASRTC_ERROR_INTERNAL_ERROR,
  ASRTC_ERROR_IO,
  ASRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
  ASRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
} asrtcResult;

/// \brief Returns a string representation of the given error code.
/// \param result [in] Error code.
/// \return Constant string describing the error.
const char *asrtcGetErrorString(asrtcResult result);

/// \brief Creates an instance of asrtcProgram with the given input parameters.
/// \param prog [out] Runtime Compilation program.
/// \param src [in] Program source.
/// \param name [in] Program name. "default_program" is used when name is ""
//                   or NULL.
/// \param numHeaders [in] Number of headers used. numHeaders must be greater
//                         than or equal to 0.
/// \param headers [in] Sources of the headers. headers should be NULL when
//                      numHeaders is 0.
/// \param includeNames [in] Name of each header by which they can be included
//                           in the CCE program source. includeNames should be
//                           NULL when numHeaders is 0. These headers must be
//                           included with the exact names specified here.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_INPUT
//   - #asrtcResult ASRTC_ERROR_IO
asrtcResult asrtcCreateProgram(asrtcProgram *prog, const char *src,
                               const char *name, int numHeaders,
                               const char *const *headers,
                               const char *const *includeNames);

/// \brief Destroys the given program.
/// \param prog [in/out] Runtime Compilation program.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
asrtcResult asrtcDestroyProgram(asrtcProgram *prog);

/// \brief Compiles the given program.
/// \param prog [in] Runtime Compilation program.
/// \param numOptions [in] Number of compiler options.
/// \param options [in] Array of option strings. options should
//                      be NULL when numOptions is 0.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
//   - #asrtcResult ASRTC_ERROR_INVALID_INPUT
//   - #asrtcResult ASRTC_ERROR_IO
//   - #asrtcResult ASRTC_ERROR_INVALID_OPTION
//   - #asrtcResult ASRTC_ERROR_COMPILE
//   - #asrtcResult ASRTC_ERROR_LINK
asrtcResult asrtcCompileProgram(asrtcProgram prog, int numOptions,
                                const char *const *options);

/// \brief Retrieves the size of the compiled device ELF binary.
/// \param prog [in] Runtime Compilation program.
/// \param deviceELFSizeRet [out] Size of the ELF binary.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
//   - #asrtcResult ASRTC_ERROR_INVALID_INPUT
asrtcResult asrtcGetDeviceELFSize(asrtcProgram prog, size_t *deviceELFSizeRet);

/// \brief Retrieves the compiled device ELF binary.
/// \param prog [in] Runtime Compilation program.
/// \param deviceELF [out] Compiled result.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
//   - #asrtcResult ASRTC_ERROR_INVALID_INPUT
asrtcResult asrtcGetDeviceELF(asrtcProgram prog, char *deviceELF);

/// \brief Notes the given name expression denoting the address of a kernel
//         function or global __gm__ variable.
/// \param prog [in] Runtime Compilation program.
/// \param nameExpression [in] Constant expression denoting the address of a
//                             kernel function or global __gm__ variable.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
asrtcResult asrtcAddNameExpression(asrtcProgram prog,
                                   const char *const nameExpression);

/// \brief Retrieves the lowered (mangled) name for a global function, The
//         memory containing the name is released when the ASRTC program is
//         destroyed by asrtcDestroyProgram or asrtcCompileProgram is called
//         again. The identical name expression must have been previously
//         provided to asrtcAddNameExpression.
/// \param prog [in] Runtime Compilation program.
/// \param nameExpression [in] Constant expression denoting the address of a
//                             kernel function or global __gm__ variable.
/// \param loweredName [out] Initialized by the function to point to a C
//                           string containing the lowered (mangled) name
//                           corresponding to the provided name expression.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
//   - #asrtcResult ASRTC_ERROR_NAME_EXPRESSION_NOT_VALID
//   - #asrtcResult ASRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
asrtcResult asrtcGetLoweredName(asrtcProgram prog, const char *nameExpression,
                                const char **loweredName);

/// \brief Retrieves the size of the compilation log.
/// \param prog [in] Runtime Compilation program.
/// \param logSizeRet [out] Size of the log string.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
//   - #asrtcResult ASRTC_ERROR_INVALID_INPUT
asrtcResult asrtcGetProgramLogSize(asrtcProgram prog, size_t *logSizeRet);

/// \brief Retrieves the compilation log.
/// \param prog [in] Runtime Compilation program.
/// \param log [out] Compilation log.
/// \return asrtcResult
//   - #asrtcResult ASRTC_SUCCESS
//   - #asrtcResult ASRTC_ERROR_INVALID_PROGRAM
//   - #asrtcResult ASRTC_ERROR_INVALID_INPUT
asrtcResult asrtcGetProgramLog(asrtcProgram prog, char *log);

#ifdef __cplusplus
}
#endif

#endif // LLVM_CLANG_ASRTC_ASRTC_H
