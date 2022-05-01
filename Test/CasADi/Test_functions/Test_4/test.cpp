//
// Created by Pietro Noah Crestaz on 12/03/22.
//

// extyract values from the sparse matrix in ptr and store in values
// which correspond to nonzeros of the sparse matrix described by
// Ir and Jc. The nonzeros of sparse matrix in ptr must be a subset of the
// nonzeros of the pattern described by Ir and Jc.
void
SparseMatrix::getValues( std::string const & func, mxArray * ptr, Number values[] ) const {

    IPOPT_DEBUG("In SparseMatrix::getValues");

    IPOPT_ASSERT(
            mxIsSparse( ptr ),
            "Error in SparseMatrix::getValues, expected sparse matrix"
    );

    // il patterm può essere un sottoinsieme
    mwIndex const * mxJc = mxGetJc(ptr);
    mwIndex const * mxIr = mxGetIr(ptr);
    double  const * v    = mxGetPr(ptr);

    std::fill_n( values, m_nnz, 0 );
    mwIndex i, k, i1, k1;
    for ( mwIndex c = 0; c < mwIndex(m_numCols); ++c ) {
        i = mxJc[c], i1 = mxJc[c+1];
        k = m_Jc[c]; k1 = m_Jc[c+1];
        for (; i < i1; ++i, ++k ) {
            mwIndex mxi = mxIr[i] ;
            while ( k < k1 && mwIndex(m_Ir[k]) < mxi ) ++k; // skip not set elements
            if ( k < k1 && mwIndex(m_Ir[k]) == mxi ) {
                IPOPT_ASSERT(
                        std::isfinite(v[i]),
                        "In MATLAB function " << func <<
                                              "\nelement (" << mxi+1 << "," << c+1 << ") is NaN\n"
                );
                values[k] = v[i];
            } else {
                IPOPT_DO_ERROR(
                        "In MATLAB function " << func <<
                                              "\nelement (" << mxi+1 << "," << c+1 << ") not found in pattern"
                );
            }
        }
    }
    IPOPT_DEBUG("Out SparseMatrix::getValues");
}