#ifndef POLYFIT_H_
#define POLYFIT_H_

#ifdef _WIN32
#ifdef SIMPLE_EXPORT
#define DLL_API __declspec(dllimport)
#else
#define DLL_API __declspec(dllexport)
#endif
#else
#define DLL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

DLL_API int polyfit(const float* const dependentValues,
                    const float* const independentValues,
                    unsigned int countOfElements, unsigned int order,
                    float* coefficients);

#ifdef __cplusplus
}
#endif

#endif