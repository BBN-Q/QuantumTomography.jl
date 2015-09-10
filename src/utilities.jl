#    Copyright 2015 Raytheon BBN Technologies
#  
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

export nearestu

"""
Computes the unitary CP map closest to a given CP map.
"""
function nearestu(l)
    c = liou2choi(l)
    vals,vecs = eig(Hermitian(c))
    imax = indmax(vals)
    Λ = mat(vecs[:,imax])
    U,Σ,V = svd(Λ)
    W = U*V'
    return kron(conj(W),W)
end
